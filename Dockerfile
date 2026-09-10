# Dockerfile
# Render's native Python runtime has no way to install system packages
# (apt.txt is a Heroku/Binder convention, not a Render feature). Docker
# gives us full control to install Tesseract (OCR), Poppler (PDF->image
# for pdf2image), and libgl1 (needed by OpenCV) - all free on Render's
# Docker runtime, same free-tier limits as the native Python runtime.

FROM python:3.13-slim

# System dependencies:
#   tesseract-ocr   - OCR engine used by pytesseract
#   poppler-utils   - provides pdftoppm/pdftocairo, required by pdf2image
#   libgl1          - required by opencv-python at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
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