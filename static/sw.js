// static/sw.js - Service Worker for offline support
const CACHE_NAME = 'ask-me-v2'; // bumped: new offline-OCR assets added
const STATIC_ASSETS = [
  '/',
  '/offline.html',
  '/static/img/ASK_ME_Logo.png',

  // Core app / offline-storage scripts
  '/static/js/csrf.js',
  '/static/js/db.js',
  '/static/js/pwa.js',

  // Offline OCR + extraction pipeline
  '/static/js/tesseract-ocr.js',
  '/static/js/extract-fields.js',
  '/static/js/offline-processor.js',

  // Tesseract.js vendor bundle (self-hosted so it can be precached — a CDN
  // script can't be relied on to be cached before the user goes offline)
  '/static/js/vendor/tesseract.min.js',
  '/static/js/vendor/tesseract-worker.min.js',
  '/static/js/vendor/tesseract-core.wasm.js',
  '/static/js/vendor/tesseract-core.wasm',
  '/static/js/vendor/pdf.min.js',
  '/static/js/vendor/pdf.worker.min.js',

  // OCR language data (English) — needed for Tesseract to actually recognize text offline
  '/static/tessdata/eng.traineddata.gz',

  'https://cdn.tailwindcss.com',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Caching static assets...');
        // Cache assets individually so one missing/renamed file (e.g. if
        // vendor files haven't been added yet) doesn't fail the whole install.
        return Promise.all(
          STATIC_ASSETS.map(url =>
            cache.add(url).catch(err => {
              console.warn('Skipping uncacheable asset:', url, err);
            })
          )
        );
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames
          .filter(name => name !== CACHE_NAME)
          .map(name => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    event.respondWith(fetch(event.request));
    return;
  }

  // Skip API calls - they need network (document processing itself now
  // happens client-side via offline-processor.js, so this only affects
  // sync/backup/admin calls, not core OCR functionality)
  if (event.request.url.includes('/api/') ||
      event.request.url.includes('/account/') ||
      event.request.url.includes('/admin/')) {
    event.respondWith(fetch(event.request).catch(() => {
      return new Response(JSON.stringify({
        error: 'Network connection required for this operation',
        offline: true
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }));
    return;
  }

  // For static assets (including OCR/vendor/lang files) - cache first
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(event.request)
          .then(response => {
            // Cache successful responses
            if (response && response.status === 200) {
              const clone = response.clone();
              caches.open(CACHE_NAME).then(cache => {
                cache.put(event.request, clone);
              });
            }
            return response;
          })
          .catch(() => {
            // Offline fallback
            const acceptHeader = event.request.headers.get('accept') || '';
            if (acceptHeader.includes('text/html')) {
              return caches.match('/offline.html');
            }
            return new Response('Offline - Please check your connection', {
              status: 503,
              statusText: 'Service Unavailable'
            });
          });
      })
  );
});

// Background Sync — fires automatically when connectivity returns, if the
// browser supports SyncManager (registered from pwa.js). Tells all open
// clients to run offlineProcessor.syncPendingDocuments().
self.addEventListener('sync', event => {
  if (event.tag === 'sync-documents') {
    event.waitUntil(
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({ type: 'RUN_DOCUMENT_SYNC' });
        });
      })
    );
  }
});

// Handle messages from the page (e.g. offline-processor.js could notify
// the SW that a document was queued, purely informational)
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'PROCESS_DOCUMENT') {
    const { id } = event.data;
    if (event.ports && event.ports[0]) {
      event.ports[0].postMessage({
        type: 'PROCESSING_QUEUED',
        id: id,
        message: 'Document queued for processing when online'
      });
    }
  }
});
