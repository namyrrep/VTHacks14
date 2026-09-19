# BLACKOUT -- offline protocol search, as one self-contained image.
#
# The image holds everything the app needs: dependencies, the embedding model, the
# prebuilt index, the UI, and the source manuals. Once built it never touches the
# network, which is the same claim the app makes on bare metal.
#
#   docker compose up          demo / submission
#   DOCKER.md                  development loop (bind mounts, no rebuild)
#
# Layer order matters: dependencies and the model are expensive and change rarely, so
# they sit above the source COPY. Editing code rebuilds only that last layer (~10s).

FROM python:3.13-slim

# HF_HOME fixes where the model cache lands so the build can bake it and the runtime
# user can read it. No bytecode: the dev overlay mounts the source read-write and we
# do not want .pyc files appearing in the working tree.
ENV HF_HOME=/opt/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Dependencies. Invalidated only when the lock file changes.
COPY backend/requirements-lock.txt ./backend/requirements-lock.txt
RUN pip install --no-cache-dir -r backend/requirements-lock.txt

# 2. The embedding model (~90 MB), downloaded once here rather than on first query.
#    This is what makes the image airplane-mode capable: search.py asks for the model
#    with local_files_only=True first, and after this layer that path succeeds.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')" \
 && chmod -R a+rX "$HF_HOME"

# 3. The application. The only layer an ordinary code change touches.
#    server.py resolves frontend/ and Documents/ as siblings of its own directory,
#    so this layout has to mirror the repo.
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY Documents/ ./Documents/

# Enforce the offline claim from here on. If the model layer above ever breaks, the
# container fails loudly at startup instead of quietly reaching for HuggingFace.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 10001 blackout
USER blackout

WORKDIR /app/backend

EXPOSE 8000

# start-period covers the index load and model warm-up that server.py does before
# it begins serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/stats').read()"

# 0.0.0.0 because the default 127.0.0.1 is unreachable from outside the container,
# and --no-browser because there is no browser in here to open.
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
