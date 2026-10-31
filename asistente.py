"""El agente: modelo + instrucciones + capabilities. Sin memoria: cada mensaje empieza de cero.

Pruébalo en la terminal, sin Telegram:  uv run asistente.py
"""

import asyncio

from pydantic_ai import Agent

from capacidades import apis, consulta, documentos, enciclopedia, registro
from config import modelo

asistente = Agent(
    modelo,
    instructions=(
        "Eres el Asistente Turístico y Guía de Viajes de Bolivia. Tu misión es orientar "
        "a los viajeros sobre destinos, atractivos turísticos, clima, rutas, historia y presupuestos "
        "en Bolivia. Responde en español, claro, cordial y en texto plano (sin Markdown). "
        "Reglas de fuentes y funciones: "
        "1. Para destinos turísticos específicos o rutas en Bolivia, revisa primero buscar_documentos y cita [fuente]. "
        "2. Si los documentos no contienen información sobre el lugar o tema consultado (o para conceptos generales, biodiversidad e historia), consulta siempre Wikipedia con resumen_wikipedia e incluye la URL al final. "
        "3. Para registrar gastos, consultar el presupuesto acumulado o eliminar/reiniciar gastos a 0, usa las tools de gastos (consultar_gastos, registrar_gasto, eliminar_gastos). "
        "4. Si ninguna fuente tiene el dato, indícalo claramente sin inventar datos."
    ),
    capabilities=[apis, consulta, registro, documentos, enciclopedia],
)


async def responder(mensaje: str) -> str:
    """Lo único que necesita el bot: un mensaje entra, una respuesta sale."""
    resultado = await asistente.run(mensaje)  # await: mientras espera al modelo, el bot atiende a otros
    return resultado.output


async def main() -> None:
    while (mensaje := input("tú> ").strip()) not in {"", "salir"}:
        print("asistente>", await responder(mensaje))


if __name__ == "__main__":
    asyncio.run(main())  # un solo asyncio.run: todo lo async vive dentro
