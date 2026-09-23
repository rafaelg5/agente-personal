"""
Ingesta del CV (PDF) y la base de conocimiento vectorial (Chroma) donde
queda indexado para que search_knowledge_base lo pueda buscar. También
incluye fetch_github_profile/fetch_gitlab_profile, que usan las tools de
agente.py para consultar esas plataformas bajo demanda (no se indexan de
antemano en Chroma).
"""

import re

import chromadb
import requests
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ---------- Ingesta de datos ----------

def load_cv_text(path: str) -> str:
    """Extrae el texto de tu CV en PDF."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_urls(text: str) -> list[str]:
    """
    Encuentra URLs dentro de un texto (ej. LinkedIn, portafolio, etc.),
    incluyendo las que no llevan http(s):// explícito (ej. "www.x.com"
    o "linkedin.com/in/usuario", formas comunes en un CV).

    Usa una whitelist de TLDs reales para evitar falsos positivos como
    "Node.js" o "React.js", y excluye dominios de email (usuario@dominio.com).
    """
    tlds = (
        "com|net|org|io|dev|co|me|info|app|ai|mx|edu|gov|us|uk|es|"
        "ca|de|fr|xyz|tech|online|site|store|cloud"
    )

    text = re.sub(r"\.\n(?=(?:" + tlds + r")\b)", ".", text)

    pattern = (
        r"(?<!@)"  # que no esté precedido de @ (evita dominios de email)
        r"\b(?:https?://)?(?:www\.)?"
        r"[a-zA-Z0-9-]+\.(?:" + tlds + r")"
        r"(?:/[^\s)>\]|,;]*)?\b"
    )
    matches = re.findall(pattern, text)

    normalized = []
    for m in matches:
        m = m.rstrip(".,;:")  # quita puntuación pegada al final
        if not m.startswith("http"):
            m = "https://" + m
        normalized.append(m)
    return list(dict.fromkeys(normalized))  # dedup preservando orden


def fetch_github_profile(username: str, token: str | None = None) -> dict:
    """Trae tu perfil, repos y READMEs de GitHub."""
    headers = {"Authorization": f"token {token}"} if token else {}
    profile = requests.get(
        f"https://api.github.com/users/{username}", headers=headers
    ).json()
    repos = requests.get(
        f"https://api.github.com/users/{username}/repos?per_page=100",
        headers=headers,
    ).json()

    repo_summaries = []
    for repo in repos:
        branch = repo.get("default_branch", "main")
        readme_url = (
            f"https://raw.githubusercontent.com/{username}/"
            f"{repo['name']}/{branch}/README.md"
        )
        r = requests.get(readme_url)
        readme = r.text[:2000] if r.status_code == 200 else ""
        repo_summaries.append(
            {
                "name": repo["name"],
                "description": repo.get("description") or "",
                "language": repo.get("language") or "",
                "stars": repo.get("stargazers_count", 0),
                "readme": readme,
            }
        )
    return {"profile": profile, "repos": repo_summaries}


def fetch_gitlab_profile(username: str, token: str | None = None) -> dict:
    """Trae tu perfil, proyectos y READMEs de GitLab."""
    headers = {"PRIVATE-TOKEN": token} if token else {}
    base_url = "https://gitlab.com/api/v4"

    users = requests.get(
        f"{base_url}/users", params={"username": username}, headers=headers
    ).json()
    profile = users[0] if users else {}
    if not profile:
        return {"profile": {}, "projects": []}

    projects = requests.get(
        f"{base_url}/users/{profile['id']}/projects",
        params={"per_page": 100},
        headers=headers,
    ).json()

    project_summaries = []
    for project in projects:
        branch = project.get("default_branch") or "main"
        path = project.get("path_with_namespace", "")
        readme_url = f"https://gitlab.com/{path}/-/raw/{branch}/README.md"
        r = requests.get(readme_url)
        readme = r.text[:2000] if r.status_code == 200 else ""
        project_summaries.append(
            {
                "name": project.get("name", ""),
                "description": project.get("description") or "",
                "stars": project.get("star_count", 0),
                "readme": readme,
            }
        )
    return {"profile": profile, "projects": project_summaries}


def format_repo(platform: str, repo: dict) -> str:
    """
    Un repo/proyecto por chunk, en texto legible en vez de JSON crudo.
    Así search_knowledge_base recupera un repo completo en un solo
    resultado, en vez de fragmentos arbitrarios de un blob JSON gigante
    partido cada 800 caracteres (lo que hacía que el modelo nunca
    encontrara la lista completa y reformulara la búsqueda sin parar).
    """
    lines = [
        f"Repositorio de {platform}: {repo.get('name', '')}",
        f"Descripción: {repo.get('description') or 'N/A'}",
    ]
    if repo.get("language"):
        lines.append(f"Lenguaje principal: {repo['language']}")
    lines.append(f"Estrellas: {repo.get('stars', 0)}")
    if repo.get("readme"):
        lines.append(f"README (extracto): {repo['readme'][:500]}")
    return "\n".join(lines)


# ---------- Vector store (base de conocimiento) ----------

class KnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # corre local, sin costo de API
        )
        self.collection = self.client.get_or_create_collection(
            name="mi_kb", embedding_function=self.embed_fn
        )

    def add_chunks(self, chunks: list[str], source: str):
        ids = [f"{source}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]
        self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    def query(self, question: str, n_results: int = 5) -> list[str]:
        results = self.collection.query(query_texts=[question], n_results=n_results)
        return results["documents"][0] if results["documents"] else []


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """
    Singleton explícito y perezoso.
    """
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def build_knowledge_base(cv_path: str = "cv.pdf") -> None:
    """
    (Re)indexa el CV en Chroma. Usa upsert internamente
    (KnowledgeBase.add_chunks), así que es seguro llamarla en cada arranque
    del proceso (ej. cada cold start en Render o Cloud Run) sin duplicar datos.

    El perfil de GitHub/GitLab ya NO se indexa aquí de antemano: el agente
    los consulta bajo demanda con las tools fetch_github_profile/
    fetch_gitlab_profile (ver tools.py) cuando encuentra una URL relevante
    en el CV o cuando el usuario da directamente un nombre de usuario.
    """
    kb = get_kb()

    cv_text = load_cv_text(cv_path)
    kb.add_chunks(chunk_text(cv_text), source="cv")

    # Las URLs del CV (LinkedIn, GitHub, GitLab, portafolio, etc.) se
    # indexan aparte
    urls = extract_urls(cv_text)
    if urls:
        kb.add_chunks(
            [f"Enlaces/URLs encontrados en el CV: {', '.join(urls)}"],
            source="cv_urls",
        )
