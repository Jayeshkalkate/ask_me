# ASK_ME - Smart Document Intelligence Chatbot

ASK_ME is an AI-powered document intelligence platform that helps users upload, process, and query documents using advanced OCR and natural language processing.

## Features

- 📄 Upload and process 26+ document types
- 🔍 OCR extraction with Tesseract and OpenBharatOCR
- 🤖 AI-powered chat interface for document queries
- ✏️ Visual and JSON data editors
- 📱 Offline-first PWA support
- 🌓 Dark/Light mode
- 🔒 Secure authentication and data isolation

## Tech Stack

- **Backend**: Django 5.1
- **Database**: PostgreSQL
- **OCR**: Tesseract, OpenCV, pdf2image
- **Background Tasks**: Celery + Redis
- **Frontend**: Tailwind CSS, JavaScript (PWA)
- **Deployment**: Render.com

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- Redis (for Celery)
- Tesseract OCR

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ask_me.git
cd ask_me