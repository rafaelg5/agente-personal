"""
Servidor HTTP para exponer agente.ask_agent() como un endpoint compatible
con la forma de respuesta de Open Responses (https://www.openresponses.org/),
pensado para conectarse desde clientes que esperan un agente autónomo
(reciben solo la respuesta final, sin function_call sin resolver).

Multi-turno: cada respuesta trae un "id"; para continuar la conversación
(que el agente recuerde de quién/qué se habló antes) manda ese id como
"previous_response_id" en el siguiente request. Sin él, cada llamada es
una conversación nueva. Ver SESSIONS más abajo para las limitaciones de
este mecanismo (en memoria, no persiste entre cold starts).

Aquí las tools (search_knowledge_base, web_search, fetch_github_profile,
fetch_gitlab_profile) se ejecutan del lado del servidor, así que
SERPAPI_API_KEY, GITHUB_TOKEN, GITLAB_TOKEN y GROQ_API_KEY nunca salen de
este proceso.

Uso local:
    uvicorn agente_server:app --host 0.0.0.0 --port 8000

En Render, Cloud Run (o cualquier host con disco efímero) el índice de Chroma no
sobrevive entre instancias/cold starts, así que el CV se reindexa cada vez
que arranca el proceso (ver build_knowledge_base en knowledge_base.py) —
es idempotente (usa upsert), solo agrega unos segundos al primer request
tras un cold start. GitHub/GitLab ya no se indexan de antemano: el agente
los consulta bajo demanda con sus propias tools.

Variables de entorno:
    AGENT_API_KEY   Token Bearer que deben mandar los clientes. En Render o
                    Cloud Run DEBE fijarse explícitamente (variables de
                    entorno del panel, o --set-env-vars en Cloud Run): si no,
                    cada instancia/cold start genera una distinta y los
                    clientes quedan con un token inválido de forma
                    intermitente.
"""

import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agente import MODEL_NAME, ask_agent
from knowledge_base import build_knowledge_base

load_dotenv()

AGENT_API_KEY = os.getenv("AGENT_API_KEY")
if not AGENT_API_KEY:
    AGENT_API_KEY = secrets.token_urlsafe(24)
    print(
        "[agente_server] AVISO: AGENT_API_KEY no definida, generada solo para "
        f"esta instancia: {AGENT_API_KEY}. En Render o Cloud Run fija esta "
        "variable explícitamente o cada cold start invalidará el token anterior."
    )

AGENT_NAME = os.getenv("AGENT_NAME") or "Agente personal (CV + GitHub + GitLab)"
AGENT_DESCRIPTION = os.getenv("AGENT_DESCRIPTION") or (
    "Responde preguntas sobre la experiencia, proyectos y perfiles de "
    "GitHub/GitLab de esta persona, usando su CV y búsqueda web cuando "
    "hace falta."
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    build_knowledge_base()
    yield


app = FastAPI(title="Agente personal (Open Responses)", lifespan=lifespan)

# Permite que clientes basados en navegador (como el formulario de "Añadir
# un agente") llamen a este endpoint directamente vía fetch(). El Bearer
# token sigue siendo la única protección real; esto solo habilita el
# preflight CORS que el navegador exige antes de la petición real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list[dict]
    previous_response_id: str | None = None


# Guarda el historial de conversación (input_items de agente.ask_agent) por
# id de respuesta, para que el cliente pueda continuar la conversación
# mandando previous_response_id — sin esto, cada request es independiente y
# el agente no sabe a quién se refiere un "su"/"él" de un turno anterior.
#
# En memoria del proceso: se pierde en cada cold start/reinicio, y en Cloud
# Run con más de una instancia un request puede caer en una instancia que
# no tiene la sesión (el balanceador no garantiza sticky sessions). Para
# el uso personal de este agente (poco tráfico, casi siempre una instancia)
# es suficiente; si se vuelve un problema real, esto necesitaría moverse a
# un store compartido (Redis, Firestore, etc.).
SESSIONS: dict[str, list[dict]] = {}


def _extract_user_question(payload: ResponsesRequest) -> str:
    if isinstance(payload.input, str):
        return payload.input
    for item in reversed(payload.input):
        if item.get("role") == "user":
            content = item.get("content")
            return content if isinstance(content, str) else str(content)
    raise HTTPException(400, "No se encontró un mensaje de usuario en 'input'.")


def _check_auth(authorization: str | None):
    if authorization != f"Bearer {AGENT_API_KEY}":
        raise HTTPException(401, "Token inválido o ausente.")


@app.post("/v1/responses")
def create_response(payload: ResponsesRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    question = _extract_user_question(payload)

    history = None
    if payload.previous_response_id:
        history = SESSIONS.get(payload.previous_response_id)
        if history is None:
            raise HTTPException(
                404,
                f"previous_response_id '{payload.previous_response_id}' no "
                "encontrado (expiró, o el proceso se reinició).",
            )

    answer, new_history = ask_agent(question, history=history)

    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    SESSIONS[response_id] = new_history

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "completed_at": int(time.time()),
        "status": "completed",
        "model": payload.model or MODEL_NAME,
        "previous_response_id": payload.previous_response_id,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": answer, "annotations": []}
                ],
            }
        ],
        "error": None,
    }


@app.get("/v1/health")
def health():
    return {"status": "ok"}


def _public_base_url(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto")
    host = request.headers.get("host", request.url.netloc)
    if not scheme:
        scheme = "http" if host.startswith(("localhost", "127.0.0.1")) else "https"
    return f"{scheme}://{host}"


@app.get("/.well-known/agent-card.json")
def agent_card(request: Request):
    responses_url = f"{_public_base_url(request)}/v1"
    return {
        "name": AGENT_NAME,
        "description": AGENT_DESCRIPTION,
        "version": "1.0.0",
        "url": f"{responses_url}/responses",
        "supportedInterfaces": [
            {
                "url": responses_url,
                "protocolBinding": "https://www.openresponses.org/specification",
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
        "security": [{"bearerAuth": []}],
        "skills": [
            {
                "id": "personal-qa",
                "name": "Preguntas sobre el perfil",
                "description": AGENT_DESCRIPTION,
                "tags": ["cv", "github", "perfil", "experiencia"],
            }
        ],
    }
