// static/js/pwa.js - PWA registration and offline support
//
// Load order required in your base template, BEFORE this file:
//   <script src="/static/js/csrf.js"></script>
//   <script src="/static/js/db.js"></script>
//   <script src="/static/js/tesseract-ocr.js"></script>
//   <script src="/static/js/extract-fields.js"></script>
//   <script src="/static/js/offline-processor.js"></script>
//   <script src="/static/js/pwa.js"></script>
//
// NOTE: db.js/offline-processor.js attach to `window.offlineStorage` /
// `window.offlineProcessor` as plain globals (not ES modules), so this file
// reads them directly instead of using `import()` — the previous version's
// `await import('./db.js')` calls never worked in a plain <script> setup.

class PWAHandler {
  constructor() {
    this.swRegistration = null;
    this.isOnline = navigator.onLine;
    this.isInstalled = false;
    this.deferredPrompt = null;

    this.init();
  }

  async init() {
    // Check if already installed
    this.isInstalled = window.matchMedia('(display-mode: standalone)').matches;

    // Register service worker
    await this.registerServiceWorker();

    // Setup online/offline listeners
    this.setupNetworkListeners();

    // Setup install prompt
    this.setupInstallPrompt();

    // Check for updates
    this.checkForUpdates();

    // Setup sync for pending operations
    this.setupBackgroundSync();

    // If we're online at startup, opportunistically sync anything left
    // over from a previous offline session.
    if (this.isOnline) {
      this.syncPendingOperations();
    }

    console.log('[PWA] Initialized successfully');
    console.log('[PWA] Installed:', this.isInstalled);
    console.log('[PWA] Online:', this.isOnline);
  }

  async registerServiceWorker() {
    try {
      if ('serviceWorker' in navigator) {
        this.swRegistration = await navigator.serviceWorker.register('/sw.js', {
          scope: '/'
        });
        console.log('[PWA] Service Worker registered');

        // Listen for messages from service worker
        navigator.serviceWorker.addEventListener('message', this.handleSWMessage.bind(this));
      }
    } catch (error) {
      console.error('[PWA] Service Worker registration failed:', error);
    }
  }

  setupNetworkListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      console.log('[PWA] Back online');
      this.syncPendingOperations();
      this.showNotification('Back online', 'Your documents will sync automatically');
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      console.log('[PWA] Offline mode');
      this.showNotification('Offline mode', 'Documents you process now will be extracted and saved on this device, and synced when you\'re back online');
    });
  }

  setupInstallPrompt() {
    window.addEventListener('beforeinstallprompt', (event) => {
      event.preventDefault();
      this.deferredPrompt = event;

      // Show install button
      this.showInstallButton();
    });

    window.addEventListener('appinstalled', () => {
      this.isInstalled = true;
      console.log('[PWA] App installed');
      this.hideInstallButton();
    });
  }

  showInstallButton() {
    const installBtn = document.getElementById('installAppBtn');
    if (installBtn) {
      installBtn.style.display = 'flex';
      installBtn.addEventListener('click', async () => {
        if (this.deferredPrompt) {
          this.deferredPrompt.prompt();
          const result = await this.deferredPrompt.userChoice;
          if (result.outcome === 'accepted') {
            console.log('[PWA] User accepted install prompt');
          }
          this.deferredPrompt = null;
          installBtn.style.display = 'none';
        }
      });
    }
  }

  hideInstallButton() {
    const installBtn = document.getElementById('installAppBtn');
    if (installBtn) {
      installBtn.style.display = 'none';
    }
  }

  handleSWMessage(event) {
    console.log('[PWA] Message from SW:', event.data);
    if (event.data.type === 'PROCESSING_QUEUED') {
      this.showNotification('Document queued', `Document ${event.data.id} will process when online`);
    }
    // Background Sync API fired in the service worker — run the actual
    // sync here in the page, where fetch + FormData + IndexedDB are available.
    if (event.data.type === 'RUN_DOCUMENT_SYNC') {
      this.syncPendingOperations();
    }
  }

  /**
   * Push any offline-processed documents (queued by offline-processor.js
   * as 'sync_document' pending ops) up to the Django backend.
   */
  async syncPendingOperations() {
    if (!this.isOnline) return;

    if (!window.offlineProcessor) {
      console.warn('[PWA] offlineProcessor not loaded — skipping sync');
      return;
    }

    try {
      const uploadUrl = '/api/offline/upload/'; // matches core/urls.py offline_upload
      const result = await window.offlineProcessor.syncPendingDocuments(uploadUrl);

      if (result.synced > 0 || result.failed > 0) {
        console.log(`[PWA] Synced ${result.synced} document(s), ${result.failed} failed`);
        if (result.synced > 0) {
          this.showNotification('Sync complete', `${result.synced} document(s) synced to your account`);
        }
      }

      // Also drain any other (non-document) pending ops, e.g. delete/update
      // operations queued while offline.
      await this.syncOtherPendingOps();
    } catch (error) {
      console.error('[PWA] Sync failed:', error);
    }
  }

  async syncOtherPendingOps() {
    if (!window.offlineStorage) return;

    const pending = await window.offlineStorage.getPendingOperations();
    const others = pending.filter(op => op.type !== 'sync_document');
    if (others.length === 0) return;

    for (const op of others) {
      try {
        await this.processOperation(op);
        await window.offlineStorage.clearPendingOperation(op.id);
      } catch (error) {
        console.error('[PWA] Sync failed for operation:', op.id, error);
        await window.offlineStorage.incrementPendingOperationRetry(op.id);
      }
    }
  }

  async processOperation(op) {
    switch (op.type) {
      case 'DELETE_DOCUMENT':
        await this.deleteDocument(op.data);
        break;
      case 'UPDATE_DOCUMENT':
        await this.updateDocument(op.data);
        break;
      default:
        console.warn('[PWA] Unknown operation type:', op.type);
    }
  }

  getCsrfToken() {
    if (window.getCsrfToken) return window.getCsrfToken(); // provided by csrf.js
    const input = document.querySelector('[name=csrfmiddlewaretoken]');
    return input ? input.value : '';
  }

  async deleteDocument(data) {
    const response = await fetch(`/document/${data.id}/delete/`, {
      method: 'POST', // matches core/urls.py delete_document (Django view, not REST DELETE)
      headers: {
        'X-CSRFToken': this.getCsrfToken()
      }
    });

    if (!response.ok) {
      throw new Error('Delete failed');
    }
    return response;
  }

  async updateDocument(data) {
    const response = await fetch('/api/offline/documents/', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': this.getCsrfToken()
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      throw new Error('Update failed');
    }
    return response.json();
  }

  setupBackgroundSync() {
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
      navigator.serviceWorker.ready.then(reg => {
        reg.sync.register('sync-documents')
          .then(() => console.log('[PWA] Background sync registered'))
          .catch(err => console.error('[PWA] Background sync registration failed:', err));
      });
    }
  }

  checkForUpdates() {
    // Check for updates periodically
    setInterval(async () => {
      if (this.swRegistration) {
        try {
          await this.swRegistration.update();
          console.log('[PWA] Checked for updates');
        } catch (error) {
          console.error('[PWA] Update check failed:', error);
        }
      }
    }, 3600000); // Check every hour
  }

  showNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(title, {
        body,
        icon: '/static/img/ASK_ME_Logo.png',
        badge: '/static/img/ASK_ME_Logo.png'
      });
    }
  }

  // Public methods
  async requestNotificationPermission() {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return false;
  }

  async getOfflineDocuments() {
    return window.offlineStorage.getAllDocuments();
  }

  async saveOfflineDocument(docData) {
    return window.offlineStorage.saveDocument(docData);
  }

  async deleteOfflineDocument(id) {
    return window.offlineStorage.deleteDocument(id);
  }

  /** Process a file fully offline: OCR + field extraction + IndexedDB save. */
  async processFileOffline(file, options) {
    if (!window.offlineProcessor) {
      throw new Error('offlineProcessor not loaded');
    }
    return window.offlineProcessor.processFile(file, options);
  }
}

// Initialize PWA
document.addEventListener('DOMContentLoaded', () => {
  const pwa = new PWAHandler();
  window.pwa = pwa;

  // Add install button to page if exists
  const installBtn = document.getElementById('installAppBtn');
  if (installBtn) {
    installBtn.style.display = 'none';
  }
});
