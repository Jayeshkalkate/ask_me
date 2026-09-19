# Matches runtime.txt (python-3.13.2) and render.yaml's `dockerfilePath: ./Dockerfile`.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# All Python deps ship manylinux wheels for slim Debian (opencv-python-headless,
# psycopg2-binary, cryptography, Pillow, numpy) - no compiler/system libs needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x entrypoint.sh

# Render (and most PaaS free tiers) inject $PORT at runtime; entrypoint.sh
# falls back to 8000 locally.
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
