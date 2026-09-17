# Online/offline + exact-answer + encryption changes

What changed in this pass, and what to do before deploying it.

## 1. Structured AI extraction (`core/ai_extract.py`)

Upload now sends Gemini the document type's *predefined field names* (from
`DOCUMENT_FIELD_TEMPLATES` in `core/models.py`) and asks for JSON
constrained to exactly those keys (`responseSchema`), with an explicit
"leave a field blank rather than guess" instruction. If that call fails for
any reason, it transparently falls back to plain OCR transcription so an
upload never hard-fails — the old regex-based extractor in `core/ai_utils.py`
then runs as a fallback over the raw text, same as before.

`core/utils.py::build_document_text_and_fields()` is the new shared merge
point used by `core/tasks.py`, `core/views.py::reprocess_document`, and
`core/views_offline.py::offline_upload` — one place instead of three copies
of the same merging logic.

**Bug fixed along the way**: `upload_document` in `views.py` was silently
throwing away the document-type dropdown; it's now saved on the `Document`
row and passed into extraction.

## 2. Exact stored-value answers (`core/field_lookup.py` + `static/js/field-lookup.js`)

`chat_api` now tries a deterministic field-name/synonym match *before* the
existing fuzzy/AI search. If your question maps to a stored field (e.g.
"what's my dob" → `Date of Birth`), you get that value back verbatim — no
LLM paraphrasing, no rounding, no guessing. Only when nothing matches
does it fall back to the old fuzzy-search + optional AI-phrased answer.

`static/js/field-lookup.js` mirrors the same matching rules in JavaScript so
offline queries (against `IndexedDB`) resolve identically with no server
call. **Keep the two `CONCEPT_SYNONYMS` tables in sync** if you edit either.

`templates/index.html`'s chat box now actually calls this: online → POST
`/api/chat/`; offline → local lookup via `window.fieldLookup` +
`window.offlineStorage`. Previously the chat form did a page-reload GET to
a different search view and had no offline path at all.

## 3. Offline upload queueing actually wired up

The upload modal in `index.html` now checks `navigator.onLine` before
uploading. Offline, it calls `window.offlineProcessor.processFile()` (from
`static/js/offline-processor.js`, now loaded globally via `base.html`) to
queue the file in IndexedDB; it gets pushed to `/api/offline/upload/` for
real extraction automatically once the browser reconnects. Previously,
uploading while genuinely offline just threw a network error.

## 4. At-rest encryption (`core/crypto_fields.py`)

`Document.extracted_text`, `extracted_data`, `user_edited_data`, and
`ai_extracted_json` are now encrypted (Fernet/AES) before hitting the
database **once you set `DOCUMENT_ENCRYPTION_KEY`**. No schema change —
existing plaintext rows keep reading correctly either way.

**To enable it:**
```bash
pip install -r requirements.txt   # adds `cryptography`
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# put the output in your .env as DOCUMENT_ENCRYPTION_KEY=...
python manage.py migrate           # applies 0003_encrypt_sensitive_fields.py (no-op on data, just field class)
```
**Back this key up like a password** — losing it makes every document
encrypted under it permanently unreadable. Without the key set, everything
behaves exactly as before (plaintext), so it's safe to skip in local dev.

Note: this covers server-side (database) at-rest encryption. Client-side
IndexedDB (offline cache) is still stored unencrypted by the browser in
this pass — reasonable next step if you need protection against local
device access, e.g. a PIN-derived key wrapping IndexedDB via WebCrypto.

## 5. Before you deploy

- `pip install -r requirements.txt` (new dependency: `cryptography`)
- Set `GEMINI_API_KEY` (unchanged requirement) and, optionally,
  `DOCUMENT_ENCRYPTION_KEY` (see above) in your `.env`
- `python manage.py migrate`
- I couldn't run `manage.py check` / a live server in this environment (no
  network/Django available here) — run it yourself once deployed:
  `python manage.py check` and `python manage.py makemigrations --check`
  (should report no changes needed).

## Known gaps not addressed in this pass

- Offline-uploaded documents are only ever persisted in the browser's
  IndexedDB, not synced into the server-side `Document` table — so they
  won't show up on another device. Fixing this means having
  `offline_upload` also create a `Document` row server-side.
- IndexedDB itself is unencrypted (see #4 above).
