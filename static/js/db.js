// static/js/db.js - IndexedDB for offline document storage
const DB_NAME = 'ask_me_db';
const DB_VERSION = 2;

class OfflineStorage {
    constructor() {
        this.db = null;
        this.isInitialized = false;
        this._initPromise = null;
    }

    async init() {
        if (this._initPromise) {
            return this._initPromise;
        }

        this._initPromise = new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = () => {
                console.error('IndexedDB error:', request.error);
                this._initPromise = null;
                reject(request.error);
            };

            request.onsuccess = () => {
                this.db = request.result;
                this.isInitialized = true;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;

                // Documents store
                if (!db.objectStoreNames.contains('documents')) {
                    const docStore = db.createObjectStore('documents', { 
                        keyPath: 'id', 
                        autoIncrement: true 
                    });
                    docStore.createIndex('user_id', 'user_id', { unique: false });
                    docStore.createIndex('created_at', 'created_at', { unique: false });
                    docStore.createIndex('doc_type', 'doc_type', { unique: false });
                    docStore.createIndex('processed', 'processed', { unique: false });
                }

                // User sessions store
                if (!db.objectStoreNames.contains('sessions')) {
                    const sessionStore = db.createObjectStore('sessions', { 
                        keyPath: 'id', 
                        autoIncrement: true 
                    });
                    sessionStore.createIndex('user_id', 'user_id', { unique: true });
                }

                // Pending operations
                if (!db.objectStoreNames.contains('pending_ops')) {
                    const opStore = db.createObjectStore('pending_ops', { 
                        keyPath: 'id', 
                        autoIncrement: true 
                    });
                    opStore.createIndex('type', 'type', { unique: false });
                    opStore.createIndex('created_at', 'created_at', { unique: false });
                }
                
                // Version 2: Add sync status index
                //
                // BUG FIX: `db.objectStore(...)` is not a real IDBDatabase
                // method (only IDBTransaction has it), so this used to throw
                // a TypeError on every single init — including the very
                // first time a user ever opened the app, since
                // event.oldVersion (0) < 2 is always true. That exception
                // inside onupgradeneeded aborts the versionchange
                // transaction, which is exactly the
                // "Version change transaction was aborted in upgradeneeded
                // event handler" error users were seeing in the upload
                // dialog. Because init() never completed, EVERY offline
                // feature was silently broken: queued offline uploads,
                // offline chat answers, and local caching all failed.
                //
                // Fix: an object store can only be reached through a
                // transaction, never through the IDBDatabase directly. By
                // this point 'documents' definitely exists (either it was
                // just created above, or it already existed from version
                // 1), so grab it via the versionchange transaction that's
                // already open for this upgrade.
                if (event.oldVersion < 2) {
                    const store = event.target.transaction.objectStore('documents');
                    if (!store.indexNames.contains('synced')) {
                        store.createIndex('synced', 'synced', { unique: false });
                    }
                }
            };
        });

        return this._initPromise;
    }

    // Document operations
    async saveDocument(docData) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['documents'], 'readwrite');
            const store = transaction.objectStore('documents');
            
            const doc = {
                ...docData,
                created_at: docData.created_at || new Date().toISOString(),
                processed: docData.processed || false,
                synced: false
            };
            
            const request = store.add(doc);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getDocument(id) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['documents'], 'readonly');
            const store = transaction.objectStore('documents');
            const request = store.get(id);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllDocuments(options = {}) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['documents'], 'readonly');
            const store = transaction.objectStore('documents');
            const request = store.getAll();

            request.onsuccess = () => {
                let docs = request.result || [];
                
                // Apply filters
                if (options.doc_type && options.doc_type !== 'all') {
                    docs = docs.filter(d => d.doc_type === options.doc_type);
                }
                if (options.status) {
                    if (options.status === 'processed') {
                        docs = docs.filter(d => d.processed === true);
                    } else if (options.status === 'pending') {
                        docs = docs.filter(d => d.processed === false);
                    } else if (options.status === 'failed') {
                        docs = docs.filter(d => d.error_message);
                    }
                }
                if (options.search) {
                    const searchLower = options.search.toLowerCase();
                    docs = docs.filter(d => 
                        (d.file_name || '').toLowerCase().includes(searchLower) ||
                        (d.extracted_text || '').toLowerCase().includes(searchLower) ||
                        (d.doc_type || '').toLowerCase().includes(searchLower)
                    );
                }
                
                // Sort by created_at descending
                docs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                resolve(docs);
            };
            request.onerror = () => reject(request.error);
        });
    }

    async updateDocument(id, data) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['documents'], 'readwrite');
            const store = transaction.objectStore('documents');
            
            const request = store.get(id);
            request.onsuccess = () => {
                const doc = request.result;
                if (!doc) {
                    reject(new Error('Document not found'));
                    return;
                }
                
                const updated = { 
                    ...doc, 
                    ...data, 
                    updated_at: new Date().toISOString(),
                    synced: false
                };
                const putRequest = store.put(updated);
                putRequest.onsuccess = () => resolve(updated);
                putRequest.onerror = () => reject(putRequest.error);
            };
            request.onerror = () => reject(request.error);
        });
    }

    async deleteDocument(id) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['documents'], 'readwrite');
            const store = transaction.objectStore('documents');
            const request = store.delete(id);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async clearAllDocuments() {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['documents'], 'readwrite');
            const store = transaction.objectStore('documents');
            const request = store.clear();

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // Session management
    async saveSession(userData) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['sessions'], 'readwrite');
            const store = transaction.objectStore('sessions');
            
            const request = store.put({
                id: 1,
                user_id: userData.id,
                username: userData.username,
                email: userData.email,
                full_name: userData.full_name || userData.get_full_name,
                profile: userData.profile || {},
                logged_in_at: new Date().toISOString()
            });

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getSession() {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['sessions'], 'readonly');
            const store = transaction.objectStore('sessions');
            const request = store.get(1);

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async clearSession() {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['sessions'], 'readwrite');
            const store = transaction.objectStore('sessions');
            const request = store.clear();

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // Pending operations
    async addPendingOperation(type, data) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['pending_ops'], 'readwrite');
            const store = transaction.objectStore('pending_ops');
            
            const request = store.add({
                type: type,
                data: data,
                created_at: new Date().toISOString(),
                retry_count: 0
            });

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getPendingOperations() {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['pending_ops'], 'readonly');
            const store = transaction.objectStore('pending_ops');
            const request = store.getAll();

            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => reject(request.error);
        });
    }

    async clearPendingOperation(id) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['pending_ops'], 'readwrite');
            const store = transaction.objectStore('pending_ops');
            const request = store.delete(id);

            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    // Increments retry_count on a pending_ops entry (NOT a document — fixes
    // the old pwa.js bug that called updateDocument() with a pending-op id).
    async incrementPendingOperationRetry(id) {
        const db = await this.init();
        return new Promise((resolve, reject) => {
            const transaction = db.transaction(['pending_ops'], 'readwrite');
            const store = transaction.objectStore('pending_ops');
            const getRequest = store.get(id);

            getRequest.onsuccess = () => {
                const op = getRequest.result;
                if (!op) {
                    resolve(null);
                    return;
                }
                op.retry_count = (op.retry_count || 0) + 1;
                const putRequest = store.put(op);
                putRequest.onsuccess = () => resolve(op);
                putRequest.onerror = () => reject(putRequest.error);
            };
            getRequest.onerror = () => reject(getRequest.error);
        });
    }
}

// Create singleton instance
const offlineStorage = new OfflineStorage();

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = offlineStorage;
} else {
    window.offlineStorage = offlineStorage;
}
