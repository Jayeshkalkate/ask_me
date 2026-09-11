// static/js/offline-processor.js
// Handles documents uploaded while the app is OFFLINE.
//
// IMPORTANT — Tesseract-based in-browser OCR has been disabled. There is
// no client-side text extraction anymore. A file uploaded while offline is
// simply saved to IndexedDB with processed=false and queued; as soon as
// the app is back online, syncPendingDocuments() pushes it to the server,
// which runs real AI extraction (Google Gemini's free tier — see
// core/ai_extract.py) and returns structured data that gets written back
// into the local record.
//
// This means: while offline, you can browse and SEARCH documents that
// were already extracted earlier (cached in IndexedDB), but a brand-new
// upload won't have its text/fields available until you're back online
// and it has synced.
//
// Load order in your template:
//   <script src="/static/js/csrf.js"></script>             <!-- Required for CSRF token -->
//   <script src="/static/js/db.js"></script>
//   <script src="/static/js/offline-processor.js"></script>

(function (global) {
    'use strict';

    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB, matches offline_upload's server limit

    /**
     * Safely retrieve the CSRF token from:
     * 1. The global `getCSRFToken()` function (provided by csrf.js)
     * 2. The cookie directly (fallback)
     */
    function getCsrfToken() {
        if (typeof global.getCSRFToken === 'function') {
            return global.getCSRFToken();
        }
        // Fallback: read from cookie
        const match = document.cookie.match(/csrftoken=([^;]+)/);
        return match ? match[1] : '';
    }

    class OfflineDocumentProcessor {
        constructor() {
            this.storage = global.offlineStorage;
        }

        async _ensureReady() {
            if (!this.storage) throw new Error('db.js (offlineStorage) not loaded.');
            if (!this.storage.isInitialized) {
                await this.storage.init();
            }
        }

        /**
         * Save a file uploaded while offline. No OCR/extraction happens
         * here anymore — the file is just stored and queued for sync.
         * Once the app is back online, syncPendingDocuments() uploads it
         * to the server for real AI-based extraction.
         *
         * @param {File} file
         * @param {Object} options { docType, userId, onProgress }
         * @returns {Promise<Object>} The saved (unprocessed) document record.
         */
        async processFile(file, options) {
            options = options || {};
            await this._ensureReady();

            if (!file) throw new Error('No file provided');
            if (file.size > MAX_FILE_SIZE) {
                throw new Error('File size must be under 10MB');
            }

            const onProgress = options.onProgress || function () {};
            onProgress('saving', 0);

            const now = new Date().toISOString();
            const documentRecord = {
                user_id: options.userId || null,
                file_name: file.name,
                file_size: file.size,
                file_blob: file, // stored directly in IndexedDB as a Blob
                doc_type: options.docType || 'other_document',
                extracted_text: '',
                extracted_data: {},
                processed: false,          // not processed yet — AI extraction runs server-side
                processed_offline: false,  // no offline OCR was performed
                pending_ai_extraction: true,
                synced: false,
                processed_at: null,
                created_at: now,
            };

            const saved = await this.storage.saveDocument(documentRecord);
            onProgress('saved', 1);

            // Queue a pending sync operation so pwa.js can push it up once online
            await this.storage.addPendingOperation('sync_document', {
                localId: saved.id || saved,
            });

            return saved;
        }

        /** Fetch all documents merging local (offline) + already-synced ones. */
        async getAllDocuments() {
            await this._ensureReady();
            return this.storage.getAllDocuments();
        }

        async getDocument(id) {
            await this._ensureReady();
            return this.storage.getDocument(id);
        }

        /** Update fields on a locally-stored document (used by the edit form). */
        async updateFields(id, updatedFields) {
            await this._ensureReady();
            const doc = await this.storage.getDocument(id);
            if (!doc) throw new Error('Document not found locally');

            doc.extracted_data = Object.assign({}, doc.extracted_data, {
                page_1: Object.assign({}, (doc.extracted_data || {}).page_1, updatedFields),
            });
            doc.synced = false; // edits need to be re-synced

            await this.storage.updateDocument(id, doc);
            await this.storage.addPendingOperation('sync_document', { localId: id });
            return doc;
        }

        async deleteDocument(id) {
            await this._ensureReady();
            await this.storage.deleteDocument(id);
        }

        /**
         * Push any pending offline documents to the server once back online.
         * The server runs real AI extraction (Gemini) and returns the
         * structured result, which we merge into the local record so it
         * becomes fully searchable offline from then on.
         *
         * @param {string} uploadUrl - The endpoint URL (e.g., '/offline-upload/')
         * @param {string} [csrfToken] - Optional; if not provided, it will be fetched
         *                               via getCsrfToken() or from the cookie.
         * @returns {Promise<Object>} { synced, failed }
         */
        async syncPendingDocuments(uploadUrl, csrfToken) {
            await this._ensureReady();
            if (!navigator.onLine) return { synced: 0, failed: 0 };

            // Ensure we have a CSRF token
            const token = csrfToken || getCsrfToken();
            const headers = {};
            if (token) {
                headers['X-CSRFToken'] = token;
            }

            const ops = await this.storage.getPendingOperations();
            let synced = 0;
            let failed = 0;

            for (const op of ops) {
                if (op.type !== 'sync_document') continue;
                try {
                    const doc = await this.storage.getDocument(op.data.localId);
                    if (!doc) {
                        await this.storage.clearPendingOperation(op.id);
                        continue;
                    }

                    const formData = new FormData();
                    if (doc.file_blob) {
                        formData.append('file', doc.file_blob, doc.file_name);
                    }
                    formData.append('doc_type', doc.doc_type);

                    const response = await fetch(uploadUrl, {
                        method: 'POST',
                        headers: headers,
                        credentials: 'same-origin', // ensure cookies are sent
                        body: formData,
                    });

                    if (response.ok) {
                        const payload = await response.json().catch(() => null);
                        const serverDoc = payload && payload.document ? payload.document : null;

                        if (serverDoc) {
                            // Merge the server's real AI-extracted data into the
                            // local record so it's searchable offline from now on.
                            doc.extracted_text = serverDoc.extracted_text || doc.extracted_text;
                            doc.extracted_data = serverDoc.extracted_data || doc.extracted_data;
                            doc.doc_type = serverDoc.doc_type || doc.doc_type;
                            doc.processed = !!serverDoc.processed;
                            doc.processed_at = serverDoc.processed_at || new Date().toISOString();
                        } else {
                            doc.processed = true;
                        }

                        doc.pending_ai_extraction = false;
                        doc.synced = true;
                        await this.storage.updateDocument(doc.id, doc);
                        await this.storage.clearPendingOperation(op.id);
                        synced++;
                    } else {
                        console.warn(`Sync failed for document ${doc.id} with status ${response.status}`);
                        failed++;
                    }
                } catch (err) {
                    console.error('Sync failed for pending op', op.id, err);
                    failed++;
                }
            }

            return { synced, failed };
        }
    }

    // Expose the processor instance globally
    global.offlineProcessor = new OfflineDocumentProcessor();
})(window);
