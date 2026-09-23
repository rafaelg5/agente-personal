FROM python:3.12-slim

WORKDIR /app

# Instala PyTorch CPU-only primero: sentence-transformers arrastra el build
# con CUDA por defecto (varios GB de paquetes nvidia-* inútiles en un host
# sin GPU como Cloud Run o Render). Fijar la versión aquí evita que pip la
# reinstale al resolver requirements.txt.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descarga el modelo de embeddings en tiempo de build para que el
# contenedor no dependa de una llamada a Hugging Face Hub en cada cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# A partir de aquí, offline: sin esto, sentence-transformers igual intenta
# un HEAD request a huggingface.co en cada arranque para revisar si hay una
# versión más nueva del modelo. La IP compartida del host (Cloud Run, Render,
# etc.) recibe 429 de Hugging Face, y la librería espera hasta 88s antes de
# reintentar — eso es lo que causaba los cold starts de más de 60s. Ya
# tenemos el modelo local.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY agente.py agente_server.py knowledge_base.py tools.py cv.pdf ./

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn agente_server:app --host 0.0.0.0 --port ${PORT}"]
