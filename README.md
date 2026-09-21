# ASK_ME — Smart Document Intelligence Assistant

ASK_ME is a Django-based document intelligence platform. Upload an ID card,
certificate, marksheet, or other document; it's read by Google Gemini's
free-tier vision API, turned into structured, searchable fields, and you can
then ask it questions in plain English ("what's my date of birth?") and get
the exact stored value back — no guessing, no paraphrasing.

It's built as an installable PWA with real offline support: documents queued
while offline sync automatically once you're back online, and previously
synced documents stay searchable with no network at all.

**Live demo:** https://ask-me-smart-document-assistant.onrender.com

## Features

- 📄 **26 predefined document types** — Aadhaar, PAN, Voter ID, Passport,
  Driving License, Vehicle RC, birth/marriage/death certificates, caste &
  income certificates, marksheets, GST certificates, invoices, and a generic
  "Other Document" catch-all — each with its own predefined field schema.
- 🤖 **AI extraction via Google Gemini** (free tier, no billing) — reads
  images and PDFs directly, no separate OCR step. For known document types it
  returns your exact predefined fields (schema-constrained JSON); for
  unrecognized documents it falls back to plain transcription plus a
  regex-based best-effort field guesser.
- 💬 **Chat interface for your documents** — a deterministic exact-field
  resolver answers direct questions ("my aadhaar number") with the literal
  stored value first; a fuzzy content search (rapidfuzz) and an optional
  Gemini-generated answer (grounded only in your own extracted data) cover
  everything else.
- 📱 **Offline-first PWA** — installable, with a service worker, IndexedDB
  local storage, and background sync. Upload while offline and it queues
  locally; ask questions offline and it searches whatever's already synced
  to this device.
- ✏️ **Visual and JSON editors** for correcting extracted data, with edits
  tracked separately from the original AI extraction.
- 🔗 **Expiring share links** for handing a single document to someone else
  without giving them account access.
- 🔒 **At-rest encryption** for extracted document data (optional, key-gated)
  and standard Django auth with per-user data isolation.
- 🌓 **Dark/light mode**, mobile-view toggle, and a lightweight admin panel
  for user management.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Django 5.1 |
| Database | PostgreSQL in production (Render/Neon), SQLite for local dev |
| Document AI | Google Gemini (`gemini-flash-latest`, free tier) via direct REST calls |
| Fuzzy search | rapidfuzz |
| File storage | Cloudinary (persists uploads across deploys — Render's free plan has no persistent disk) |
| Background processing | Python `threading` (Celery is scaffolded in settings but currently disabled — see [Known limitations](#known-limitations)) |
| Frontend | Django templates, Tailwind CSS, vanilla JS |
| Offline / PWA | Service worker + IndexedDB (`static/js/db.js`, `offline-processor.js`, `pwa.js`) |
| Static files | WhiteNoise |
| Deployment | Render.com (`render.yaml`, native Python buildpack — no Dockerfile) |

> **Note on OCR:** this project used to run Tesseract + OpenCV locally
> (`core/ocr_utils.py` still contains that pipeline). It's currently disabled
> in favor of sending documents straight to Gemini, which needs no system
> binary on the server and generally reads messy phone-camera scans better.
> The old code is still there behind an `if pytesseract:` guard for anyone
> who wants to re-enable a fully offline/no-API-key extraction path — see
> [Re-enabling local OCR](#re-enabling-local-ocr-optional).

## How extraction works

1. You upload a file and (optionally) pick its document type.
2. If the type has a predefined field schema (`core/models.py ->
   DOCUMENT_FIELD_TEMPLATES`), Gemini is asked to return **exactly those
   keys** as JSON, with an explicit instruction to leave a field blank
   rather than invent a value.
3. If that structured call fails for any reason (rate limit, transient
   outage, blocked prompt), it falls back to a plain transcription-only
   prompt — you still get the raw text and full-text search still works,
   but structured fields are intentionally left empty rather than guessed,
   and the document is flagged for reprocessing once the AI service
   recovers.
4. Only for genuinely unrecognized ("Other Document") uploads does a
   regex-based heuristic (`core/ai_utils.py`) attempt to guess fields like
   Name, DOB, or address from the raw text.

## Installation

### Prerequisites

- Python 3.10+ (Render deploys on 3.13.2 — see `runtime.txt`)
- PostgreSQL (optional locally — SQLite is used automatically if
  `DATABASE_URL` isn't set)
- A free [Google AI Studio](https://aistudio.google.com/apikey) API key
  (for document extraction and chat)
- A free [Cloudinary](https://cloudinary.com) account (for persistent file
  storage — optional locally, required in production)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ask_me.git
   cd ask_me/chatbot/ask_me
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and fill it in:
   ```bash
   cp .env.example .env
   ```
   At minimum, set `SECRET_KEY` and `GEMINI_API_KEY`. See
   [Environment variables](#environment-variables) below for the full list.

5. Run migrations and create a superuser:
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. Collect static files (needed once, or whenever static assets change):
   ```bash
   python manage.py collectstatic --noinput
   ```

7. Run the dev server:
   ```bash
   python manage.py runserver
   ```
   Visit `http://127.0.0.1:8000`.

## Environment variables

All read via `.env` (loaded with `python-dotenv`) or real environment
variables in production. None are required just to boot the app locally —
sensible defaults keep `runserver`/`collectstatic` working — but extraction,
chat, email, and persistent storage won't work without their respective keys.

| Variable | Required | Purpose |
|---|---|---|
| `SECRET_KEY` | Production | Django secret key. Render can auto-generate this. |
| `DEBUG` | No | `True`/`False`. Defaults to `False`. |
| `ALLOWED_HOSTS` | Production | Comma-separated hostnames. |
| `CSRF_TRUSTED_ORIGINS` | Production | Comma-separated origins, each including `https://`. |
| `TIME_ZONE` | No | Defaults to `Asia/Kolkata`. |
| `DATABASE_URL` | No (SQLite fallback) | Postgres connection string. |
| `GEMINI_API_KEY` | **Yes** | Powers document extraction (`core/ai_extract.py`) and AI chat (`core/ai_chat.py`). Free, no card required, from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `CLOUDINARY_URL` | Production | Persistent file storage. Without this, uploaded files are written to local disk and **will be lost** on the next deploy/restart on Render's free tier. Get it from your Cloudinary dashboard → API Environment variable. |
| `DOCUMENT_ENCRYPTION_KEY` | No | Encrypts extracted document data at rest. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Leave blank in dev. **Losing this key permanently locks you out of already-encrypted rows** — store it like a password, never in source control. |
| `EMAIL_HOST_USER` / `EMAIL_HOST_PASSWORD` | No | Gmail SMTP + App Password, used for password-reset emails. |

## Deployment (Render)

`render.yaml` is already set up for Render's free tier, native Python
buildpack (no Dockerfile):

```yaml
buildCommand: pip install -r requirements.txt && python manage.py collectstatic --noinput
startCommand: python manage.py migrate --noinput && gunicorn ask_me.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
```

`SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, and `TIME_ZONE` are set directly in
`render.yaml`. You must set the rest — `DATABASE_URL`, `GEMINI_API_KEY`,
`CLOUDINARY_URL`, `DOCUMENT_ENCRYPTION_KEY`, `CSRF_TRUSTED_ORIGINS`,
`EMAIL_HOST_USER`/`EMAIL_HOST_PASSWORD` — in the Render dashboard's
Environment tab, not in the YAML file.

**Don't skip `CLOUDINARY_URL` in production.** Render's free plan wipes local
disk on every deploy and restart; without Cloudinary configured, every
uploaded document is silently lost the next time the service redeploys or
spins down.

## Project structure

```
chatbot/ask_me/
├── account/            # Custom auth: login/register/password reset, admin user list
├── ask_me/             # Project settings, root URLconf
├── core/               # Main app
│   ├── models.py       #   Document model + DOCUMENT_FIELD_TEMPLATES (the 26 schemas)
│   ├── ai_extract.py   #   Gemini-based structured/generic extraction
│   ├── ai_chat.py      #   Gemini-based chat answer generation (grounded in your docs)
│   ├── ai_utils.py     #   Regex-based fallback field detection (unknown doc types only)
│   ├── field_lookup.py #   Deterministic "exact field value" resolver for chat
│   ├── ocr_utils.py    #   Legacy Tesseract/OpenCV pipeline (disabled by default)
│   ├── crypto_fields.py#   Optional at-rest encryption for extracted data
│   ├── views.py        #   Upload, document detail/edit/reprocess, chat API, sharing
│   ├── views_offline.py#   Sync endpoint for documents queued while offline
│   └── tasks.py         #   Background extraction (thread-based)
├── static/js/
│   ├── db.js            #   IndexedDB wrapper
│   ├── offline-processor.js  # Queues/syncs documents saved while offline
│   ├── field-lookup.js  #   JS mirror of field_lookup.py, for offline chat answers
│   └── pwa.js           #   Service worker registration + background sync
├── templates/           # Server-rendered HTML (Tailwind)
├── render.yaml          # Render deployment config
└── requirements.txt
```

## Known limitations

- **Background processing uses a plain Python thread**, not a real task
  queue. Celery/Redis config is scaffolded in `settings.py` but commented
  out. This is fine for light traffic, but if the gunicorn worker restarts
  mid-extraction, that document is left stuck as unprocessed. Worth moving
  to Celery + Redis (or Render's own background workers) if usage grows.
- **Local OCR (Tesseract) is disabled.** All extraction depends on Gemini's
  free tier being reachable; if Google's API has an outage, uploads still
  succeed but structured fields won't populate until you reprocess.
- **Gemini's free tier has rate limits.** Under heavy use you may see
  transient 503s — the app retries once automatically and otherwise flags
  the document for a manual "Reprocess."

## Re-enabling local OCR (optional)

If you want a genuine offline/no-API-key fallback for when Gemini is down,
`core/ocr_utils.py` already contains a full Tesseract + OpenCV + OpenBharatOCR
pipeline behind optional imports — it's just never called by the current
upload flow. To bring it back you'd need to:

1. Add `pytesseract`, `pdf2image`, and `openbharatocr` back to
   `requirements.txt`.
2. Install the underlying system binaries (`tesseract-ocr`, `poppler-utils`)
   — Render's native Python buildpack can't install non-Python system
   packages, so this requires switching to a Dockerfile-based deploy.
3. Wire `process_document_file_enhanced()` back in as a fallback in
   `core/tasks.py` / `core/views_offline.py` when `extract_document_ai()`
   returns an error.

## Security notes

- `.gitignore` already excludes `.env` and `db.sqlite3`, but both were
  previously committed to this repo's history. If this repository (or its
  history) is public, **rotate `GEMINI_API_KEY` and `SECRET_KEY`**, then
  remove both files from git history (`git rm --cached .env db.sqlite3` at
  minimum going forward; use `git filter-repo` or BFG to scrub history).
- Set `DOCUMENT_ENCRYPTION_KEY` in production if you're storing real
  identity documents — it encrypts extracted Aadhaar/PAN/DOB/address data at
  rest with no schema changes required.

## License

NA