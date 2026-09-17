# core/crypto_fields.py
"""
Transparent at-rest encryption for sensitive Document fields (Aadhaar/PAN/
DOB/address/etc. live inside extracted_data & extracted_text).

Design goals:
  - No schema/migration change. EncryptedJSONField still stores its value in
    a normal JSONField column - it just stores a single encrypted string
    instead of the raw dict once a key is configured. EncryptedTextField
    stores an encrypted string in the existing TextField column.
  - Backward compatible. Rows written before DOCUMENT_ENCRYPTION_KEY was
    configured are plain dicts/text already sitting in the DB. Reading them
    back simply returns the value as-is (decryption is tried first; if it
    fails because the value was never encrypted, we fall through instead of
    raising, so existing data keeps working).
  - Opt-in but on by default once configured. If DOCUMENT_ENCRYPTION_KEY is
    unset, both fields behave exactly like the stock Django fields they
    subclass - nothing breaks in dev/test environments that haven't set a
    key yet.

Setup: generate a key once with
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
and set DOCUMENT_ENCRYPTION_KEY in your .env. Losing this key makes
previously-encrypted documents unrecoverable - back it up like a password.
"""
import json
import logging

from django.conf import settings
from django.db import models

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover - guarded so the app still boots
    Fernet = None
    InvalidToken = Exception


def _get_fernet():
    """Return a configured Fernet instance, or None if encryption is off."""
    if Fernet is None:
        return None
    key = getattr(settings, "DOCUMENT_ENCRYPTION_KEY", "") or ""
    if not key:
        return None
    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except (ValueError, TypeError):
        logger.error(
            "DOCUMENT_ENCRYPTION_KEY is set but is not a valid Fernet key - "
            "storing data UNENCRYPTED. Generate one with "
            "Fernet.generate_key()."
        )
        return None


class EncryptedJSONField(models.JSONField):
    """A JSONField whose value is encrypted at rest when a key is configured."""

    def get_prep_value(self, value):
        fernet = _get_fernet()
        if fernet is None or not value:
            return super().get_prep_value(value)
        try:
            plaintext = json.dumps(value, default=str).encode("utf-8")
            token = fernet.encrypt(plaintext).decode("utf-8")
            return super().get_prep_value(token)
        except (TypeError, ValueError):
            logger.exception("Failed to encrypt JSON field value; storing as-is")
            return super().get_prep_value(value)

    def from_db_value(self, value, expression, connection):
        value = super().from_db_value(value, expression, connection)
        fernet = _get_fernet()
        if fernet is None or not isinstance(value, str):
            return value
        try:
            plaintext = fernet.decrypt(value.encode("utf-8"))
            return json.loads(plaintext)
        except (InvalidToken, ValueError, TypeError):
            # Legacy plaintext row from before encryption was enabled, or no
            # key configured when it was written - hand it back unchanged.
            return value

    def to_python(self, value):
        # Called for values coming from forms/fixtures rather than the DB.
        if isinstance(value, str):
            fernet = _get_fernet()
            if fernet is not None:
                try:
                    plaintext = fernet.decrypt(value.encode("utf-8"))
                    return json.loads(plaintext)
                except (InvalidToken, ValueError, TypeError):
                    pass
        return super().to_python(value)


class EncryptedTextField(models.TextField):
    """A TextField whose value is encrypted at rest when a key is configured."""

    def get_prep_value(self, value):
        fernet = _get_fernet()
        value = super().get_prep_value(value)
        if fernet is None or not value:
            return value
        try:
            return fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        except (TypeError, ValueError):
            logger.exception("Failed to encrypt text field value; storing as-is")
            return value

    def from_db_value(self, value, expression, connection):
        fernet = _get_fernet()
        if fernet is None or not isinstance(value, str) or not value:
            return value
        try:
            return fernet.decrypt(value.encode("utf-8")).decode("utf-8")
        except (InvalidToken, ValueError, TypeError):
            return value


def encryption_enabled() -> bool:
    """Whether DOCUMENT_ENCRYPTION_KEY is configured and usable right now."""
    return _get_fernet() is not None
