# Agente personal (CV + GitHub + GitLab)

Agente conversacional que responde preguntas sobre una persona específica usando su CV, y bajo demanda sus perfiles de GitHub/GitLab y búsqueda web. Implementa el protocolo abierto [Open Responses](https://www.openresponses.org/), corriendo gratis sobre [Groq](https://console.groq.com).

## Cómo funciona

- **Modelo**: Groq (`openai/gpt-oss-120b` por defecto), vía `client.responses.create()` del SDK de OpenAI apuntando a `https://api.groq.com/openai/v1`.
- **Base de conocimiento**: tu CV (PDF), indexado en [ChromaDB](https://www.trychroma.com/) con embeddings locales (`sentence-transformers`). Las URLs que aparecen en el CV (LinkedIn, GitHub, GitLab, portafolio, etc.) se indexan aparte para que el agente las encuentre fácilmente.
- **Tools**: `search_knowledge_base` (busca en Chroma), `fetch_github_profile`/`fetch_gitlab_profile` (consultan esas plataformas en vivo cuando el agente encuentra una URL relevante en el CV o el usuario da directamente un nombre de usuario) y `web_search` (SerpAPI). Todas se ejecutan del lado del servidor.
- **Servidor**: FastAPI (`agente_server.py`) expone el agente como un endpoint HTTP compatible con Open Responses, protegido con un token Bearer propio.
- **Despliegue**: Docker + Google Cloud Run (escala a cero, gratis en reposo). El índice de Chroma se reconstruye en cada cold start (el disco no persiste entre instancias), no se necesita almacenamiento externo.

## Estructura del proyecto

| Archivo | Qué contiene |
|---|---|
| `agente.py` | El loop del agente (`ask_agent`): llama al modelo, ejecuta tool calls, repite hasta tener respuesta final. Único punto (junto con `agente_server.py`) que lee variables de entorno. |
| `knowledge_base.py` | Ingesta y clase `KnowledgeBase` sobre Chroma (solo el CV se indexa). También trae `fetch_github_profile`/`fetch_gitlab_profile` (usadas por `tools.py` para consultas en vivo). No lee variables de entorno — recibe todo como parámetros. |
| `tools.py` | Definición de las tools (`TOOLS`) y su implementación (`TOOL_FUNCTIONS`): búsqueda en la KB, GitHub, GitLab y web. Tampoco lee variables de entorno directamente — `agente.py` le pasa las keys/tokens vía `configure()`. |
| `agente_server.py` | Servidor FastAPI: expone `/v1/responses`, `/v1/health` y `/.well-known/agent-card.json`; maneja auth y CORS. |
| `Dockerfile` | Imagen para desplegar en Cloud Run (PyTorch CPU-only, modelo de embeddings pre-descargado, `HF_HUB_OFFLINE=1`). |
| `.env.example` | Template de variables de entorno — cópialo a `.env` y completa los valores. |

## Requisitos previos

1. Cuenta gratis en [console.groq.com](https://console.groq.com) → genera una API key.
2. Cuenta gratis en [serpapi.com](https://serpapi.com) → genera una API key.
3. Python 3.12+ (el proyecto se probó también en 3.14).

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# completa GROQ_API_KEY y SERPAPI_API_KEY en .env
```

## Uso local

**Indexar el CV en Chroma:**
```bash
python agente.py
```

**Probar una pregunta directamente en Python:**
```bash
python -c "import agente; agente.build_knowledge_base(); r, _ = agente.ask_agent('¿Qué experiencia tiene esta persona?'); print(r)"
```
`build_knowledge_base()` es necesario aquí — sin él, `ask_agent()` busca contra lo que ya haya en `chroma_db/` de una corrida anterior (o nada, si es la primera vez). Si ya corriste `python agente.py` antes en esta sesión de shell y no cambiaste `cv.pdf`, puedes omitirlo.

**Levantar el servidor HTTP:**
```bash
uvicorn agente_server:app --host 0.0.0.0 --port 8000
```
Al arrancar imprime un `AGENT_API_KEY` generado si no definiste uno fijo en `.env`.

**Probarlo:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <AGENT_API_KEY>" \
  -d '{"input": "¿Qué experiencia tiene esta persona?"}'
```

**Conversación de varios turnos:** cada respuesta trae un `id`. Mándalo como `previous_response_id` en el siguiente request para que el agente recuerde de qué/quién se venía hablando (sin esto, cada llamada es una conversación nueva):
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <AGENT_API_KEY>" \
  -d '{"input": "¿Tiene perfiles en redes sociales?", "previous_response_id": "resp_xxxxx"}'
```
El historial se guarda en memoria del proceso — se pierde al reiniciar el servidor, y en Cloud Run con más de una instancia un `previous_response_id` puede no encontrarse si el request cae en otra instancia (devuelve 404).

## Variables de entorno

Ver [.env.example](.env.example) para la lista completa con defaults. Las únicas dos obligatorias son `GROQ_API_KEY` y `SERPAPI_API_KEY`.

## Despliegue en Cloud Run

Requiere el CLI de `gcloud` autenticado y un proyecto de GCP con facturación habilitada (el uso normal de este agente cae dentro del free tier).

```bash
gcloud run deploy agente-personal \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars "GROQ_API_KEY=...,SERPAPI_API_KEY=...,GITHUB_TOKEN=...,GITLAB_TOKEN=...,AGENT_API_KEY=..."
```

`--allow-unauthenticated` es correcto aquí: el acceso lo controla el `AGENT_API_KEY` propio del servidor, no IAM de Cloud Run. **Fija `AGENT_API_KEY` explícitamente** — si no, cada cold start genera uno nuevo y los clientes quedan con un token inválido de forma intermitente.

## Conectar el agente a otras herramientas

`agente_server.py` expone `/.well-known/agent-card.json` (tarjeta de agente estilo A2A) con la URL de Open Responses, para que herramientas que soportan "importar agente desde tarjeta" lo detecten automáticamente. La clave de API (`AGENT_API_KEY`) no viaja en la tarjeta — hay que pegarla manualmente donde corresponda.

## Seguridad

- `GROQ_API_KEY`, `SERPAPI_API_KEY`, `GITHUB_TOKEN` y `GITLAB_TOKEN` viven solo como variables de entorno del servidor — nunca se envían al cliente ni al modelo.
- `AGENT_API_KEY` es la única credencial que ve el exterior; protege `/v1/responses` (todo lo demás, incluida la tarjeta de agente, es público sin auth).
- `.env` está en `.gitignore` — nunca lo subas al repo. Usa `.env.example` como referencia.
