"""
Tools que el agente puede invocar: búsqueda en la base de conocimiento local
(CV), consulta bajo demanda de perfiles de GitHub/GitLab, y búsqueda web
(SerpAPI). Incluye las definiciones en formato Responses API (TOOLS) y el
despachador de nombre -> función (TOOL_FUNCTIONS).
"""

import json

import serpapi

from knowledge_base import fetch_github_profile, fetch_gitlab_profile, format_repo, get_kb


def tool_search_kb(query: str) -> str:
    docs = get_kb().query(query)
    return "\n---\n".join(docs) if docs else "No se encontró información relevante."


_github_token: str | None = None
_gitlab_token: str | None = None


def tool_fetch_github_profile(username: str) -> str:
    data = fetch_github_profile(username, token=_github_token)
    if not data.get("profile", {}).get("login"):
        return f"No se encontró un usuario de GitHub llamado '{username}'."
    lines = [f"Perfil de GitHub: {json.dumps(data['profile'], ensure_ascii=False)}"]
    lines += [format_repo("GitHub", r) for r in data["repos"]]
    return "\n---\n".join(lines)


def tool_fetch_gitlab_profile(username: str) -> str:
    data = fetch_gitlab_profile(username, token=_gitlab_token)
    if not data.get("profile"):
        return f"No se encontró un usuario de GitLab llamado '{username}'."
    lines = [f"Perfil de GitLab: {json.dumps(data['profile'], ensure_ascii=False)}"]
    lines += [format_repo("GitLab", p) for p in data["projects"]]
    return "\n---\n".join(lines)


class WebSearch:
    def __init__(self, api_key: str):
        self.client = serpapi.Client(api_key=api_key)

    def query(self, query: str) -> list[str]:
        try:
            results = self.client.search({
                "engine": "google",
                "q": query,
                "google_domain": "google.com.mx",
                "hl": "es",
                "gl": "mx",
                "num": 10
            })
        except Exception:
            # No relanzar: la excepción del cliente de SerpAPI incluye la URL
            # completa de la petición (con la api_key en el query string) en
            # su mensaje, y eso terminaría en el traceback de los logs.
            return "La búsqueda web no está disponible en este momento."

        organic = results.get("organic_results", [])
        if not organic:
            return "No se encontraron resultados web para esta búsqueda."

        formatted = []
        for r in organic:
            formatted.append(
                f"Título: {r.get('title', '')}\n"
                f"Fuente: {r.get('link', '')}\n"
                f"Resumen: {r.get('snippet', '')}"
            )
        return "\n---\n".join(formatted)


_ws: WebSearch | None = None


def configure(
    serpapi_api_key: str,
    github_token: str | None = None,
    gitlab_token: str | None = None,
) -> None:
    """
    Debe llamarse una vez al arrancar (desde agente.py, que es quien lee
    las variables de entorno) antes de usar las tools de este módulo.
    github_token/gitlab_token son opcionales (solo suben el rate limit de
    esas APIs); fetch_github_profile/fetch_gitlab_profile funcionan sin
    ellos.
    """
    if not serpapi_api_key:
        raise RuntimeError(
            "Falta SERPAPI_API_KEY: pásala a tools.configure(serpapi_api_key)."
        )
    global _ws, _github_token, _gitlab_token
    _ws = WebSearch(api_key=serpapi_api_key)
    _github_token = github_token
    _gitlab_token = gitlab_token


def tool_web_search(query: str) -> str:
    """
    Placeholder: conecta aquí una API de búsqueda real, por ejemplo
    Tavily, Brave Search API o SerpAPI, y devuelve resultados en texto.
    """
    if _ws is None:
        raise RuntimeError("tools.configure(serpapi_api_key) no fue llamado.")
    return _ws.query(query)


TOOLS = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": (
            "Busca información en tu CV (experiencia, proyectos, stack "
            "técnico, educación, y URLs encontradas en el documento)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Busca información pública y actual sobre ti en internet. "
            "También puedes pasar directamente una URL (ej. un perfil "
            "de LinkedIn encontrado en la base de conocimiento) como "
            "query para obtener lo que esté indexado públicamente "
            "sobre esa página."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "fetch_github_profile",
        "description": (
            "Obtiene el perfil y los repositorios públicos de un usuario de "
            "GitHub (descripción, lenguaje, estrellas y extracto del README "
            "de cada repo). Úsala cuando encuentres una URL de GitHub en el "
            "CV/base de conocimiento, o cuando el usuario te dé directamente "
            "un nombre de usuario de GitHub."
        ),
        "parameters": {
            "type": "object",
            "properties": {"username": {"type": "string"}},
            "required": ["username"],
        },
    },
    {
        "type": "function",
        "name": "fetch_gitlab_profile",
        "description": (
            "Obtiene el perfil y los proyectos públicos de un usuario de "
            "GitLab (descripción, estrellas y extracto del README de cada "
            "proyecto). Úsala cuando encuentres una URL de GitLab en el "
            "CV/base de conocimiento, o cuando el usuario te dé directamente "
            "un nombre de usuario de GitLab."
        ),
        "parameters": {
            "type": "object",
            "properties": {"username": {"type": "string"}},
            "required": ["username"],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_knowledge_base": tool_search_kb,
    "web_search": tool_web_search,
    "fetch_github_profile": tool_fetch_github_profile,
    "fetch_gitlab_profile": tool_fetch_gitlab_profile,
}
