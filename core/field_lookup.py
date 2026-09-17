# core/field_lookup.py
"""
Deterministic "exact field value" resolver for chatbot queries.

Problem this solves: the existing rapidfuzz-based matching in views.py is a
similarity SCORE, not an answer - and the AI-chat layer (ai_chat.py) then
paraphrases whatever it matched, which is a regeneration step sitting
between the user and the actual stored value. For a question like "what is
my date of birth", the requirement is to return the exact stored value with
no rewriting, no rounding, no guessing.

This module answers that narrower question first: given a user's message
and a document's structured field dict (predefined keys like "Date of
Birth", "Aadhaar Number" - see models.DOCUMENT_FIELD_TEMPLATES), find the
single field the question is asking about and return its value completely
unchanged. If nothing matches with confidence, it returns None and the
caller falls back to the existing fuzzy/AI-assisted search - which is a
different, looser feature (searching document *content*, not answering a
specific-field question).

IMPORTANT: static/js/field-lookup.js mirrors this logic in JavaScript for
offline mode (no server round-trip). Keep the two in sync if you change the
synonym table or matching rules below.
"""
import re
from typing import Dict, List, Optional, Tuple

# Concept -> phrases a user might type when asking about a field whose name
# contains that concept. Matching is bidirectional: a concept "hits" when
# ANY of its phrases appears in the question AND the (normalized) field name
# contains that same concept word.
CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "number": ["number", "no", "num", "id"],
    "name": ["name", "full name", "called"],
    "date of birth": ["dob", "date of birth", "birth date", "born", "birthday"],
    "gender": ["gender", "sex"],
    "mobile": ["mobile", "phone", "contact number", "cell number", "phone number"],
    "address": ["address", "residence", "residential address", "location", "live"],
    "father": ["father", "father's name", "fathers name", "dad"],
    "mother": ["mother", "mother's name", "mothers name", "mom"],
    "husband": ["husband", "husband's name"],
    "wife": ["wife", "wife's name"],
    "expiry": ["expiry", "expiration", "valid until", "validity", "expires"],
    "issue": ["issue date", "date of issue", "issued on", "when issued"],
    "nationality": ["nationality", "citizen of"],
    "blood group": ["blood group", "blood type"],
    "vehicle": ["vehicle number", "registration number", "reg number"],
    "engine": ["engine number", "engine no"],
    "chassis": ["chassis number", "chassis no", "vin"],
    "category": ["category", "caste"],
    "income": ["income", "annual income", "salary"],
    "percentage": ["percentage", "marks", "score"],
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse to single spaces."""
    if not text:
        return ""
    words = _WORD_RE.findall(text.lower())
    return " ".join(words)


def _concept_words(field_key_norm: str) -> List[str]:
    """Which concepts (from CONCEPT_SYNONYMS) does this normalized field name contain?"""
    hits = []
    for concept in CONCEPT_SYNONYMS:
        concept_norm = normalize(concept)
        if concept_norm and concept_norm in field_key_norm:
            hits.append(concept)
    return hits


def resolve_field(question: str, available_fields: List[str]) -> Optional[str]:
    """
    Given a user's question and the list of field keys actually present on a
    document, return the single field key the question is most specifically
    asking about, or None if there's no confident match.

    Matching, in order of confidence:
      1. The full normalized field name appears verbatim in the question
         (e.g. field "Aadhaar Number", question "what's my aadhaar number").
      2. A concept synonym match: the field name contains a known concept
         (e.g. "date of birth") and the question contains one of that
         concept's phrasings (e.g. "dob").
    Ties are broken by preferring the longer/more specific field name, so
    "Aadhaar Number" beats a bare "Number" field if both are somehow present.
    """
    if not question or not available_fields:
        return None

    q_norm = normalize(question)
    if not q_norm:
        return None

    exact_hits = []
    concept_hits = []

    for field_key in available_fields:
        key_norm = normalize(field_key)
        if not key_norm:
            continue

        # 1. Direct containment of the field name in the question.
        if key_norm in q_norm:
            exact_hits.append(field_key)
            continue

        # 2. Concept-based synonym match.
        for concept in _concept_words(key_norm):
            for phrase in CONCEPT_SYNONYMS[concept]:
                if phrase in q_norm:
                    concept_hits.append(field_key)
                    break

    if exact_hits:
        exact_hits.sort(key=len, reverse=True)
        return exact_hits[0]

    if concept_hits:
        # Deduplicate while preserving first-seen order, then prefer the
        # most specific (longest) field name among the matches.
        seen = list(dict.fromkeys(concept_hits))
        seen.sort(key=len, reverse=True)
        return seen[0]

    return None


def get_exact_field_answer(question: str, fields: Dict[str, object]) -> Optional[Tuple[str, object]]:
    """
    fields: a flat {field_key: value} dict for ONE document (e.g. from
    Document.display_data, internal keys already stripped by the caller).
    Returns (field_key, value) for the best match, or None.
    """
    if not fields:
        return None
    usable = {k: v for k, v in fields.items() if v not in (None, "", [])}
    if not usable:
        return None

    match = resolve_field(question, list(usable.keys()))
    if match is None:
        return None
    return match, usable[match]
