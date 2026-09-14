"""
Agente RAG personal: responde preguntas sobre ti usando tu CV, y bajo
demanda tu GitHub/GitLab y búsqueda web.

Usa la API gratuita de Groq (https://console.groq.com) para correr el modelo,
hablándole con el protocolo abierto Open Responses (https://www.openresponses.org/),
que Groq expone de forma nativa y compatible con client.responses.create()
del SDK de OpenAI. No requiere GPU propia ni un servidor corriendo 24/7.

La ingesta de datos y la base de conocimiento viven en knowledge_base.py;
las tools del agente viven en tools.py. Este archivo es solo el loop del
agente en sí.

Requisitos previos:
    1. Crea una cuenta gratis en https://console.groq.com y genera una API key.
    2. Defínela como GROQ_API_KEY en tu .env

Dependencias de Python:
    pip install openai chromadb sentence-transformers pypdf requests python-dotenv

Variables de entorno:
    GROQ_API_KEY (requerida)
    SERPAPI_API_KEY (requerida, para la tool de búsqueda web)
    GITHUB_TOKEN (opcional, sube el rate limit de la API de GitHub)
    GITLAB_TOKEN (opcional, sube el rate limit de la API de GitLab)
    LLM_BASE_URL (opcional, por defecto https://api.groq.com/openai/v1)
    MODEL_NAME (opcional, por defecto openai/gpt-oss-120b)
    MODEL_FALLBACK_CHAIN (opcional, lista separada por comas; por defecto
        "MODEL_NAME,openai/gpt-oss-20b" — si un modelo agota su cupo diario
        en Groq (RateLimitError), se prueba el siguiente de la lista)
    TEMPERATURE (opcional, 0-2, por defecto 1)
    TOP_P (opcional, 0-1, por defecto 1)
    MAX_TOOL_ITERATIONS (opcional, por defecto 4)
    MAX_OUTPUT_TOKENS (opcional, por defecto 4096)

Todas las variables de entorno se leen únicamente aquí y en
agente_server.py — knowledge_base.py y tools.py reciben todo como
parámetros explícitos (github_token, gitlab_token, serpapi_api_key, etc.),
sin tocar os.getenv() ni load_dotenv() por su cuenta.
"""

import json
import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

from knowledge_base import build_knowledge_base
from tools import TOOL_FUNCTIONS, TOOLS
from tools import configure as configure_tools

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "Falta la variable de entorno GROQ_API_KEY (crea una gratis en "
        "https://console.groq.com y defínela en .env)."
    )

configure_tools(
    serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
    github_token=os.getenv("GITHUB_TOKEN"),
    gitlab_token=os.getenv("GITLAB_TOKEN"),
)

# Cliente Open Responses: apunta a la API gratuita de Groq en vez de un
# servidor local de Ollama, así que ya no depende de tu GPU/CPU ni de que
# tu máquina esté prendida.
# Nota: "os.getenv(X) or default" en vez de "os.getenv(X, default)" en todo
# este bloque — si la variable existe pero vacía (ej. siguiendo el
# .env.example sin llenarla), la segunda forma devuelve "" en vez del
# default, y rompe los int()/float() de abajo.
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b"
# Groq limita tokens por día (TPD) por modelo, no por cuenta — si MODEL_NAME
# se queda sin cupo, probamos el siguiente de esta lista antes de rendirnos.
# Solo gpt-oss-120b/20b: qwen3.6-27b/qwen3.8-27b se probaron y no sirven aquí
# (gastan su presupuesto de salida entero en razonamiento interno sin llegar
# a responder, por su límite de 1000 tokens de salida por minuto en Groq).
# groq/compound tampoco: no soporta tools propias.
_fallback_raw = os.getenv("MODEL_FALLBACK_CHAIN") or f"{MODEL_NAME},openai/gpt-oss-20b"
MODEL_FALLBACK_CHAIN = list(dict.fromkeys(m.strip() for m in _fallback_raw.split(",") if m.strip()))
TEMPERATURE = float(os.getenv("TEMPERATURE") or "1")
TOP_P = float(os.getenv("TOP_P") or "1")
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS") or "4")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS") or "4096")

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre una persona
específica usando su CV, su GitHub y GitLab, y búsquedas web cuando sea
necesario. Usa siempre search_knowledge_base primero para fundamentar tu
respuesta. Si no encuentras información suficiente, dilo honestamente en
vez de inventar datos.

Sobre GitHub/GitLab: si search_knowledge_base te devuelve una URL de GitHub
o GitLab (o el usuario te da directamente un nombre de usuario de alguna de
estas plataformas), usa fetch_github_profile o fetch_gitlab_profile con ese
usuario para traer sus repositorios/proyectos reales antes de responder.
Si no hay información de la cuenta de GitHub o GitLab en el CV y el usuario
tampoco te dio un nombre de usuario, pídeselo en vez de adivinarlo.

Sobre web_search: úsala solo si la pregunta requiere información actual o
externa que el CV/GitHub/GitLab no puedan tener (ej. noticias recientes).
Si search_knowledge_base ya te dio una respuesta completa, NO la verifiques
con web_search — respondes directo. Si de todos modos usas web_search y no
obtienes nada útil, no lo repitas con variaciones de la misma búsqueda:
responde con lo que ya tengas de search_knowledge_base."""


def ask_agent(
    user_question: str, history: list[dict] | None = None
) -> tuple[str, list[dict]]:
    """
    Devuelve (respuesta, historial_actualizado). Pasa el historial que te
    devolvió la llamada anterior para que el agente recuerde de quién se
    viene hablando (ej. "busca sus redes sociales" refiriéndose a alguien
    mencionado en un turno previo). Sin historial, arranca una conversación
    nueva con solo el system prompt.

    Quien mantiene el historial entre requests HTTP separados es
    agente_server.py (vía previous_response_id) — esta función es agnóstica
    a cómo se persiste.
    """
    if history:
        input_items = list(history) + [{"role": "user", "content": user_question}]
    else:
        input_items = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = None
        last_rate_limit_error = None
        for model in MODEL_FALLBACK_CHAIN:
            try:
                response = client.responses.create(
                    model=model,
                    input=input_items,
                    tools=TOOLS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )
                if model != MODEL_FALLBACK_CHAIN[0]:
                    print(f"[agente] usando fallback '{model}' (modelo(s) anterior(es) sin cupo)")
                break
            except openai.RateLimitError as e:
                last_rate_limit_error = e
                continue
            except (openai.APIStatusError, openai.APIConnectionError):
                return (
                    "Tuve un problema técnico al procesar tu pregunta. "
                    "Intenta reformularla o vuelve a intentar en un momento."
                ), (history or [])

        if response is None:
            # Se agotó el cupo en todos los modelos de MODEL_FALLBACK_CHAIN.
            return (
                "Se alcanzó el límite de uso en todos los modelos disponibles "
                f"por ahora ({last_rate_limit_error.message}). Intenta de "
                "nuevo más tarde."
            ), (history or [])

        input_items += [item.model_dump() for item in response.output]

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not function_calls:
            text = "".join(
                block.text
                for item in response.output
                if item.type == "message"
                for block in item.content
                if block.type == "output_text"
            )
            if response.status == "incomplete":
                note = "\n\n[Respuesta cortada por el límite de tokens configurado (MAX_OUTPUT_TOKENS).]"
                final_text = (text + note) if text else (
                    "La respuesta se cortó por el límite de tokens antes de "
                    "generar texto visible. Sube MAX_OUTPUT_TOKENS o haz la "
                    "pregunta más específica."
                )
            else:
                final_text = text
            return final_text, input_items

        for call in function_calls:
            args = json.loads(call.arguments)
            result = TOOL_FUNCTIONS[call.name](**args)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                }
            )

    return (
        "No pude terminar de investigar tras varios pasos de búsqueda. "
        "Intenta una pregunta más específica."
    ), input_items


if __name__ == "__main__":
    build_knowledge_base()
