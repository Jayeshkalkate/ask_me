// static/js/offline-processor.js
// Ties together tesseract-ocr.js (OCR), extract-fields.js (structured
// extraction) and db.js (offlineStorage / IndexedDB) into one pipeline
// that works with zero network connectivity, mirroring the shape of the
// data the Django views (`views_offline.py`) already produce.
//
// Load order in your template:
//   <script src="/static/js/vendor/tesseract.min.js"></script>
//   <script src="/static/js/vendor/pdf.min.js"></script>   (optional, for PDFs)
//   <script src="/static/js/db.js"></script>
//   <script src="/static/js/tesseract-ocr.js"></script>
//   <script src="/static/js/extract-fields.js"></script>
//   <script src="/static/js/offline-processor.js"></script>

(function (global) {
    'use strict';

    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB, matches offline_upload's server limit

    class OfflineDocumentProcessor {
        constructor() {
            this.storage = global.offlineStorage;
            this.ocr = global.offlineOCR;
            this.extract = global.offlineExtract;
        }

        async _ensureReady() {
            if (!this.storage) throw new Error('db.js (offlineStorage) not loaded.');
            if (!this.ocr) throw new Error('tesseract-ocr.js (offlineOCR) not loaded.');
            if (!this.extract) throw new Error('extract-fields.js (offlineExtract) not loaded.');
            if (!this.storage.isInitialized) {
                await this.storage.init();
            }
        }

        /**
         * Full offline pipeline for a single uploaded file:
         *   1. OCR the file entirely in-browser (no network)
         *   2. Run structured field extraction on the OCR text
         *   3. Save everything into IndexedDB
         *   4. Return the same document shape the app's UI already expects
         *
         * @param {File} file
         * @param {Object} options { docType, userId, onProgress }
         */
        async processFile(file, options) {
            options = options || {};
            await this._ensureReady();

            if (!file) throw new Error('No file provided');
            if (file.size > MAX_FILE_SIZE) {
                throw new Error('File size must be under 10MB');
            }

            const onProgress = options.onProgress || function () {};
            onProgress('ocr_start', 0);

            // --- 1. OCR ---
            const ocrResult = await this.ocr.processFile(file, (status, progress) => {
                onProgress('ocr_progress', progress, status);
            });
            onProgress('ocr_done', 1);

            const combinedText = this.extract.cleanOcrText(ocrResult.combinedText);

            // --- 2. Structured extraction ---
            onProgress('extract_start', 0);
            let extractedData = {};
            const perPageData = {};
            ocrResult.pages.forEach((page) => {
                const pageFields = this.extract.extractStructuredData(page.rawText);
                if (Object.keys(pageFields).length > 0) {
                    perPageData[`page_${page.page}`] = pageFields;
                }
            });
            extractedData = Object.keys(perPageData).length > 0
                ? perPageData
                : { page_1: this.extract.extractStructuredData(combinedText) };
            onProgress('extract_done', 1);

            // --- Determine doc type ---
            let docType = options.docType || 'other_document';
            if (!docType || docType === 'other_document') {
                const detected = this.extract.detectDocumentType(combinedText);
                if (detected && detected !== 'other_document') {
                    docType = detected;
                }
            }

            // --- 3. Build document record (same shape as offline_upload's JSON) ---
            const now = new Date().toISOString();
            const documentRecord = {
                user_id: options.userId || null,
                file_name: file.name,
                file_size: file.size,
                file_blob: file, // stored directly in IndexedDB as a Blob
                doc_type: docType,
                extracted_text: combinedText.slice(0, 5000),
                extracted_data: extractedData,
                processed: true,
                processed_offline: true,
                synced: false, // will flip to true once pushed to the server
                processed_at: now,
                created_at: now,
            };

            // --- 4. Save to IndexedDB ---
            const saved = await this.storage.saveDocument(documentRecord);
            onProgress('saved', 1);

            // Queue a pending sync operation so pwa.js can push it up later
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
         * Call this from pwa.js on the 'online' event.
         */
        async syncPendingDocuments(uploadUrl) {
            await this._ensureReady();
            if (!navigator.onLine) return { synced: 0, failed: 0 };

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

                    const csrfToken = global.getCsrfToken ? global.getCsrfToken() : '';
                    const response = await fetch(uploadUrl, {
                        method: 'POST',
                        headers: csrfToken ? { 'X-CSRFToken': csrfToken } : {},
                        body: formData,
                    });

                    if (response.ok) {
                        doc.synced = true;
                        await this.storage.updateDocument(doc.id, doc);
                        await this.storage.clearPendingOperation(op.id);
                        synced++;
                    } else {
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

    global.offlineProcessor = new OfflineDocumentProcessor();
})(window);
