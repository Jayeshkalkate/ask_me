# core/constants.py

MIN_TEXT_LENGTH_FOR_DETECTION = 20
MIN_TEXT_LENGTH_FOR_EXTRACTION = 30

# For AI chat context
MAX_CONTEXT_DOCS = 3
MAX_TEXT_SNIPPET_CHARS = 500

# API timeout
REQUEST_TIMEOUT = 12  # seconds

# Gemini model
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Internal keys to skip when building field dictionaries
INTERNAL_KEYS = {"_metadata", "_page", "_type"}   # adjust as needed