# Dockerfile
# Render's native Python runtime has no way to install system packages
# (apt.txt is a Heroku/Binder convention, not a Render feature). Docker
# gives us full control to install libgl1 (needed by OpenCV) - free on
# Render's Docker runtime, same free-tier limits as the native Python
# runtime.
#
# Tesseract + Poppler were removed: text extraction now uses Google
# Gemini's free API tier (see core/ai_extract.py) instead of on-server
# Tesseract OCR, so those system binaries are no longer needed.

FROM python:3.13-slim

# System dependencies:
#   libgl1          - required by opencv-python at import time (image
#                      quality/blur checks, unrelated to text extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
# separately from your application code.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application.
COPY . .

# Copy entrypoint and make it executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Render sets $PORT at runtime; entrypoint.sh binds gunicorn to it
# (falls back to 8000 locally when PORT is not set).
ENTRYPOINT ["./entrypoint.sh"]