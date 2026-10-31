import truststore
truststore.inject_into_ssl()
import time
"""Las capabilities de tu asistente: lo que vimos en clase. Cámbialas o agrega las tuyas.

Una Capability junta tools + instrucciones en una pieza que se enchufa al agente (asistente.py).
El docstring y los tipos de cada tool son lo único que el modelo lee para decidir cuándo usarla.
"""

import csv
import sqlite3
from datetime import date
from functools import cache
from typing import Literal

import chromadb
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_ai.capabilities import Capability
from pypdf import PdfReader

from config import DATOS, settings

# ═══ S1 · Tools de lectura con APIs públicas ══════════════════════════════════════════
apis = Capability(
    id="apis_publicas",
    description="Consulta indicadores económicos (Banco Mundial) y clima (Open Meteo) con sus APIs públicas",
    instructions="Usa las APIs para obtener datos reales. Si una API no tiene el dato, dilo, y no inventes datos",
)


@apis.tool_plain
def inflacion(pais: str, anio: int) -> dict:
    """Inflación anual de un país en un año, según el Banco Mundial.

    Args:
        pais: código ISO del país, por ejemplo 'BO' para Bolivia o 'PE' para Perú
        anio: año, por ejemplo 2024
    """
    url = f"https://api.worldbank.org/v2/country/{pais}/indicator/FP.CPI.TOTL.ZG"
    respuesta = httpx.get(url, params={"date": anio, "format": "json"}, timeout=10)
    respuesta.raise_for_status()
    datos = respuesta.json()

    # La API responde [metadatos, registros]. Si no hay dato, devolvemos un aviso que el modelo puede leer
    if len(datos) < 2 or not datos[1] or datos[1][0]["value"] is None:
        return {"error": f"No hay datos de inflación disponibles para {pais} en {anio}"}

    registro = datos[1][0]
    return {
        "pais": registro["country"]["value"],
        "anio": registro["date"],
        "inflacion_pct": round(registro["value"], 2),
        "fuente": "Banco Mundial",  # devolver la fuente evita que el modelo invente una
    }


@apis.tool_plain
def clima_cochabamba() -> dict:
    """Temperatura actual de Cochabamba, según Open Meteo."""
    params = {"latitude": -17.38, "longitude": -66.15, "current": "temperature_2m"}
    respuesta = httpx.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    respuesta.raise_for_status()
    return {"temperatura": respuesta.json()["current"]["temperature_2m"], "fuente": "Open Meteo"}


# ═══ S1 · Mi contador: gastos en SQLite, una tool que lee y otra que escribe ══════════
Categoria = Literal["transporte", "alimentacion", "educacion", "ocio", "salud", "hospedaje"]
CSV = DATOS / "gastos.csv"  # 10 gastos ficticios de mayo de 2030 para pruebas
DB = DATOS / "gastos.db"  # la base de datos de gastos SQLite


def crear_base(cargar_ejemplos: bool = False) -> None:
    """Crea gastos.db. Solo carga datos del CSV si cargar_ejemplos=True o si se ejecuta dentro de pytest."""
    DB.unlink(missing_ok=True)
    import sys
    filas = []
    if (cargar_ejemplos or "pytest" in sys.modules) and CSV.exists():
        with CSV.open(encoding="utf-8") as archivo:
            filas = [(g["fecha"], g["categoria"], g["descripcion"], float(g["monto"])) for g in csv.DictReader(archivo)]
    conexion = sqlite3.connect(DB)
    with conexion:
        conexion.execute(
            "CREATE TABLE gastos (fecha TEXT, categoria TEXT, descripcion TEXT, monto REAL CHECK (monto > 0))"
        )
        if filas:
            conexion.executemany("INSERT INTO gastos VALUES (?, ?, ?, ?)", filas)
    conexion.close()


def hoy() -> str:
    """Instrucción dinámica: el modelo no sabe qué día es; se lo decimos en cada mensaje."""
    return f"Hoy es {date.today().isoformat()}."


# Lectura y escritura van en capabilities separadas: a otro agente le podrías dar solo la primera
consulta = Capability(
    id="consulta_gastos",
    description="Consulta gastos personales y presupuesto almacenados en una base de datos SQLite",
    instructions="Usa la base de datos para responder preguntas sobre los gastos por categoría o el total general. No inventes datos.",
)
registro = Capability(
    id="registro_gastos",
    description="Registra nuevos gastos o elimina gastos en la base de datos SQLite para reiniciar el presupuesto",
    instructions=["Permite agregar gastos o eliminarlos para reiniciar el presupuesto a 0 cuando el usuario lo solicite.", hoy],
)


@consulta.tool_plain
def consultar_gastos(categoria: Categoria | None = None) -> dict:
    """Gastos registrados y total en bolivianos.

    Args:
        categoria: Una categoría ('transporte', 'alimentacion', 'hospedaje', etc.)
                   o None para ver todos los gastos y el total acumulado.
    """
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)  # solo lectura
    conexion.row_factory = sqlite3.Row  # cada fila como diccionario
    if categoria:
        filas = conexion.execute(
            "SELECT fecha, categoria, descripcion, monto FROM gastos WHERE categoria = ?",
            (categoria,),
        ).fetchall()
        conexion.close()
        return {
            "categoria": categoria,
            "total_bs": sum(f["monto"] for f in filas),
            "cantidad": len(filas),
            "gastos": [dict(f) for f in filas],
        }
    else:
        filas = conexion.execute(
            "SELECT fecha, categoria, descripcion, monto FROM gastos ORDER BY fecha DESC"
        ).fetchall()
        conexion.close()
        resumen_cat = {}
        for f in filas:
            cat = f["categoria"]
            resumen_cat[cat] = resumen_cat.get(cat, 0.0) + f["monto"]
        return {
            "categoria": "todas",
            "total_bs": sum(f["monto"] for f in filas),
            "cantidad_total": len(filas),
            "por_categoria": resumen_cat,
            "gastos": [dict(f) for f in filas],
        }


@registro.tool_plain
def registrar_gasto(fecha: str, categoria: Categoria, descripcion: str, monto: float) -> str:
    """Anota un gasto nuevo del presupuesto. La fecha va como AAAA-MM-DD y el monto en bolivianos."""
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB)  # sin mode=ro: esta tool sí escribe
    try:
        with conexion:  # transacción: guarda todo o nada
            conexion.execute("INSERT INTO gastos VALUES (?, ?, ?, ?)", (fecha, categoria, descripcion, monto))
    except sqlite3.IntegrityError:  # el CHECK de la tabla rechaza montos de 0 o negativos
        return f"No se guardó: el monto debe ser mayor que 0 (recibí {monto})."
    finally:
        conexion.close()
    return f"Gasto registrado: {descripcion}, {monto} Bs en {categoria} el {fecha}."


@registro.tool_plain
def eliminar_gastos(categoria: Categoria | None = None) -> str:
    """Elimina gastos para reiniciar el presupuesto a 0 o borrar una categoría específica.

    Args:
        categoria: Categoría a borrar ('transporte', 'alimentacion', 'hospedaje', etc.)
                   o None si se desea borrar TODOS los gastos y reiniciar a 0.
    """
    if not DB.exists():
        crear_base()
    conexion = sqlite3.connect(DB)
    try:
        with conexion:
            if categoria:
                cursor = conexion.execute("DELETE FROM gastos WHERE categoria = ?", (categoria,))
                cant = cursor.rowcount
                return f"Se eliminaron {cant} gasto(s) de la categoría '{categoria}'."
            else:
                cursor = conexion.execute("DELETE FROM gastos")
                cant = cursor.rowcount
                return f"Se eliminaron todos los gastos registrados ({cant} registros). El presupuesto está reiniciado en 0 Bs."
    finally:
        conexion.close()


# ═══ S2 · RAG: buscar en mis documentos (PDFs) y citar la fuente ══════════════════════
DOCUMENTOS = DATOS / "documentos"  # PDFs ficticios de finanzas personales


def embeddings_de(textos: list[str]) -> list[list[float]]:
    """Convierte cada texto en un vector: textos de significado parecido dan vectores cercanos."""
    # Gemini también ofrece una API compatible con la de OpenAI; la usamos para los embeddings
    cliente = OpenAI(
        api_key=settings.gemini_api_key_local.get_secret_value(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    respuesta = cliente.embeddings.create(model="gemini-embedding-001", input=textos)
    return [item.embedding for item in respuesta.data]


CHROMA_DIR = DATOS / "chroma_db"


@cache  # se indexa una sola vez de forma persistente en disco
def indice():
    # Base vectorial persistente: si ya está indexada, responde al instante sin gastar cuota de Gemini
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coleccion = cliente.get_or_create_collection(
        "documentos", configuration={"hnsw": {"space": "cosine"}}
    )
    if coleccion.count() > 0:
        return coleccion

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    for pdf in sorted(DOCUMENTOS.glob("*.pdf")):
        try:
            texto = "\n".join(pagina.extract_text() or "" for pagina in PdfReader(pdf).pages)
        except Exception:
            continue
        if not texto.strip():
            continue
        trozos = splitter.split_text(texto)
        if not trozos:
            continue
        lote_tamano = 10
        for i in range(0, len(trozos), lote_tamano):
            lote = trozos[i : i + lote_tamano]
            coleccion.upsert(
                ids=[f"{pdf.stem}-{i + j}" for j in range(len(lote))],
                documents=lote,
                embeddings=embeddings_de(lote),
            )
            time.sleep(2.0)
    return coleccion


def buscar(pregunta: str, k: int = 3) -> list[dict]:
    """Los k trozos más parecidos a la pregunta, con su fuente y su similitud (1 = idéntico)."""
    r = indice().query(query_embeddings=embeddings_de([pregunta]), n_results=k)
    return [
        {"fuente": fuente, "similitud": round(1 - distancia, 2), "texto": texto}
        for fuente, distancia, texto in zip(r["ids"][0], r["distances"][0], r["documents"][0])
    ]


documentos = Capability(
    id="guias_turismo",
    description="Guías turísticas y destinos de Bolivia: atractivos, rutas, cultura y recomendaciones",
    instructions=(
        "Para preguntas sobre destinos turísticos de Bolivia, atractivos, rutas o lugares usa buscar_documentos "
        "y cita la fuente entre corchetes, por ejemplo [Guia-Turistica-de-Robore-1]. "
        "Si la información en los documentos no cubre el lugar o tema consultado, recurre a resumen_wikipedia."
    ),
)


@documentos.tool_plain
def buscar_documentos(pregunta: str) -> list[dict]:
    """Busca en los documentos los fragmentos que responden a la pregunta."""
    resultados = buscar(pregunta, k=3)
    for r in resultados:  # en la terminal (o en los Logs de Render) se ve qué recuperó
        print(f"  [DOC] {r['fuente']} (similitud {r['similitud']})", flush=True)
    return resultados


# ═══ S2 · Wikipedia: fuente enciclopédica externa ══════════════════════════════════════
enciclopedia = Capability(
    id="enciclopedia",
    description="Resúmenes enciclopédicos de Wikipedia en español sobre ciudades, monumentos e historia",
    instructions=(
        "Para historia general, conceptos o datos enciclopédicos sobre lugares de Bolivia usa resumen_wikipedia. "
        "Termina citando la fuente con su URL."
    ),
)


@enciclopedia.tool_plain
async def resumen_wikipedia(tema: str) -> dict:
    """Resumen corto de un artículo de Wikipedia en español, con su URL."""
    from urllib.parse import quote
    headers = {"User-Agent": "MiAsistenteTuristico/1.0 (contacto@turismo-bolivia.com)"}
    tema_limpio = tema.strip().strip("¿?¡!").strip()
    async with httpx.AsyncClient(timeout=15, headers=headers) as cliente:
        # Búsqueda con la API de búsqueda de Wikipedia para resolver títulos reales aun con artículos o prefijos
        url_search = f"https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(tema_limpio)}&utf8=1&format=json"
        titulo = tema_limpio
        try:
            r_search = await cliente.get(url_search)
            if r_search.status_code == 200:
                resultados = r_search.json().get("query", {}).get("search", [])
                if resultados:
                    titulo = resultados[0]["title"]
        except Exception:
            pass

        print(f"  [WIKI] {tema} -> {titulo}", flush=True)

        url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{quote(titulo.replace(' ', '_'))}"
        r = await cliente.get(url)
        if r.status_code != 200:
            return {"error": f"No se encontró información en Wikipedia para '{tema}'"}
        datos = r.json()
        return {
            "tema": datos.get("title", titulo),
            "resumen": datos.get("extract", "Sin extracto disponible"),
            "url": datos.get("content_urls", {}).get("desktop", {}).get("page", f"https://es.wikipedia.org/wiki/{quote(titulo.replace(' ', '_'))}"),
            "fuente": "Wikipedia",
        }
