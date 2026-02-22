# C:\chatbot\ask_me\Dockerfile

# Use lightweight python
FROM python:3.11-slim

# Avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (IMPORTANT for OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy full project
COPY . .

# Expose port
EXPOSE 8000

# Run django
CMD ["gunicorn", "ask_me.wsgi:application", "--bind", "0.0.0.0:8000"]