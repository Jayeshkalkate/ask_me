// static/js/field-lookup.js
// Mirrors core/field_lookup.py so offline chat queries (no server) resolve
// a question to an exact stored field value the same way the online
// /api/chat/ endpoint does. Keep the CONCEPT_SYNONYMS table below in sync
// with the Python version if you edit either one.
(function (global) {
    'use strict';

    const CONCEPT_SYNONYMS = {
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
    };

    const INTERNAL_KEYS = new Set(["_metadata", "status", "raw_text", "structured_data", "fields", "Content"]);

    function normalize(text) {
        if (!text) return "";
        const words = String(text).toLowerCase().match(/[a-z0-9]+/g);
        return words ? words.join(" ") : "";
    }

    function conceptWords(fieldKeyNorm) {
        const hits = [];
        for (const concept in CONCEPT_SYNONYMS) {
            const conceptNorm = normalize(concept);
            if (conceptNorm && fieldKeyNorm.indexOf(conceptNorm) !== -1) {
                hits.push(concept);
            }
        }
        return hits;
    }

    /**
     * @param {string} question
     * @param {string[]} availableFields
     * @returns {string|null} the best-matching field key, or null
     */
    function resolveField(question, availableFields) {
        const qNorm = normalize(question);
        if (!qNorm || !availableFields || !availableFields.length) return null;

        const exactHits = [];
        const conceptHits = [];

        for (const fieldKey of availableFields) {
            const keyNorm = normalize(fieldKey);
            if (!keyNorm) continue;

            if (qNorm.indexOf(keyNorm) !== -1) {
                exactHits.push(fieldKey);
                continue;
            }

            const concepts = conceptWords(keyNorm);
            let matched = false;
            for (const concept of concepts) {
                for (const phrase of CONCEPT_SYNONYMS[concept]) {
                    if (qNorm.indexOf(phrase) !== -1) {
                        conceptHits.push(fieldKey);
                        matched = true;
                        break;
                    }
                }
                if (matched) break;
            }
        }

        if (exactHits.length) {
            exactHits.sort((a, b) => b.length - a.length);
            return exactHits[0];
        }
        if (conceptHits.length) {
            const seen = Array.from(new Set(conceptHits));
            seen.sort((a, b) => b.length - a.length);
            return seen[0];
        }
        return null;
    }

    /**
     * Flatten a document's stored extracted_data/user_edited_data (which is
     * shaped like {page_1: {field: value, ...}, ...}) into one {field: value}
     * map, skipping internal keys - mirrors utils.INTERNAL_KEYS filtering.
     */
    function flattenDocumentFields(doc) {
        const data = (doc && (doc.extracted_data || {})) || {};
        const fields = {};
        for (const pageKey in data) {
            const pageData = data[pageKey];
            if (!pageData || typeof pageData !== 'object') continue;
            for (const key in pageData) {
                if (INTERNAL_KEYS.has(key) || key.startsWith('_')) continue;
                const value = pageData[key];
                if (value === null || value === undefined || value === '') continue;
                if (!(key in fields)) fields[key] = value;
            }
        }
        return fields;
    }

    /**
     * Try to answer `question` from an array of locally-stored documents
     * (as returned by offlineStorage.getAllDocuments()).
     *
     * @returns {{docId: *, docType: string, field: string, value: *}|null}
     */
    function answerFromDocuments(question, documents) {
        if (!question || !documents || !documents.length) return null;

        let best = null;
        for (const doc of documents) {
            if (!doc.processed) continue;
            const fields = flattenDocumentFields(doc);
            const fieldKeys = Object.keys(fields);
            if (!fieldKeys.length) continue;

            const match = resolveField(question, fieldKeys);
            if (match) {
                const candidate = {
                    docId: doc.id,
                    docType: doc.doc_type || 'other_document',
                    field: match,
                    value: fields[match],
                };
                // Prefer the most recently created/updated matching document.
                if (!best || new Date(doc.created_at || 0) > new Date(best._created || 0)) {
                    best = Object.assign({ _created: doc.created_at }, candidate);
                }
            }
        }

        if (!best) return null;
        delete best._created;
        return best;
    }

    global.fieldLookup = {
        normalize,
        resolveField,
        flattenDocumentFields,
        answerFromDocuments,
    };
})(window);
