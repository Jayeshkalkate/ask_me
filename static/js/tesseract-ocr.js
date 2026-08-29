// static/js/tesseract-ocr.js
// Fully offline, in-browser OCR using Tesseract.js (WASM).
// No network calls once the Tesseract core/worker/lang files are cached
// by the service worker (see sw.js STATIC_ASSETS).
//
// Requires tesseract.min.js to be loaded on the page BEFORE this file, e.g.:
//   <script src="/static/js/vendor/tesseract.min.js"></script>
//   <script src="/static/js/tesseract-ocr.js"></script>

(function (global) {
    'use strict';

    // Local paths so everything is served from your own domain and can be
    // pre-cached by the service worker for true offline use.
    const WORKER_PATH = '/static/js/vendor/tesseract-worker.min.js';
    const CORE_PATH = '/static/js/vendor/tesseract-core.wasm.js';
    const LANG_PATH = '/static/tessdata/'; // expects eng.traineddata(.gz) here

    class OfflineOCR {
        constructor() {
            this.worker = null;
            this._readyPromise = null;
        }

        /**
         * Lazily create and initialize the Tesseract worker.
         * Safe to call multiple times — reuses the same worker.
         */
        async _ensureWorker(onProgress) {
            if (this.worker) {
                return this.worker;
            }
            if (this._readyPromise) {
                return this._readyPromise;
            }

            this._readyPromise = (async () => {
                if (typeof Tesseract === 'undefined') {
                    throw new Error(
                        'Tesseract.js is not loaded. Include tesseract.min.js before tesseract-ocr.js.'
                    );
                }

                const worker = await Tesseract.createWorker('eng', 1, {
                    workerPath: WORKER_PATH,
                    corePath: CORE_PATH,
                    langPath: LANG_PATH,
                    logger: (msg) => {
                        if (onProgress && msg.status && typeof msg.progress === 'number') {
                            onProgress(msg.status, msg.progress);
                        }
                    },
                });

                this.worker = worker;
                return worker;
            })();

            return this._readyPromise;
        }

        /**
         * Run OCR on a File/Blob (image) or a canvas element.
         * Returns { rawText, confidence } — same shape the rest of the app
         * expects from the old server-side `raw_text` field.
         */
        async recognizeImage(fileOrCanvas, onProgress) {
            const worker = await this._ensureWorker(onProgress);
            const result = await worker.recognize(fileOrCanvas);
            return {
                rawText: result.data.text || '',
                confidence: result.data.confidence || 0,
            };
        }

        /**
         * Run OCR on every page of a PDF using pdf.js to rasterize pages to
         * canvases first (pdf.js must also be loaded and cached offline).
         * Returns an array of { page, rawText, confidence }.
         */
        async recognizePdf(file, onProgress) {
            if (typeof pdfjsLib === 'undefined') {
                throw new Error('pdf.js is not loaded — cannot process PDF offline.');
            }

            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            const pages = [];

            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);
                const viewport = page.getViewport({ scale: 2.0 }); // ~ similar to 300dpi
                const canvas = document.createElement('canvas');
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                const ctx = canvas.getContext('2d');
                await page.render({ canvasContext: ctx, viewport }).promise;

                const { rawText, confidence } = await this.recognizeImage(
                    canvas,
                    onProgress
                );
                pages.push({ page: pageNum, rawText, confidence });
            }

            return pages;
        }

        /**
         * Convenience entry point: figures out if the file is an image or
         * a PDF and returns a normalized result:
         *   { pages: [{ page, rawText, confidence }, ...], combinedText }
         */
        async processFile(file, onProgress) {
            const isPdf =
                file.type === 'application/pdf' ||
                file.name.toLowerCase().endsWith('.pdf');

            let pages;
            if (isPdf) {
                pages = await this.recognizePdf(file, onProgress);
            } else {
                const { rawText, confidence } = await this.recognizeImage(
                    file,
                    onProgress
                );
                pages = [{ page: 1, rawText, confidence }];
            }

            const combinedText = pages.map((p) => p.rawText).join('\n');
            return { pages, combinedText };
        }

        /** Simple lightweight image pre-processing (grayscale + contrast)
         *  before OCR, similar in spirit to the cv2 steps in ocr_utils.py. */
        async preprocessImage(file) {
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);

                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data;
                    for (let i = 0; i < data.length; i += 4) {
                        const gray =
                            0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                        // simple contrast stretch
                        const contrasted = Math.min(255, Math.max(0, (gray - 128) * 1.3 + 128));
                        data[i] = data[i + 1] = data[i + 2] = contrasted;
                    }
                    ctx.putImageData(imageData, 0, 0);
                    resolve(canvas);
                };
                img.onerror = reject;
                img.src = URL.createObjectURL(file);
            });
        }

        async terminate() {
            if (this.worker) {
                await this.worker.terminate();
                this.worker = null;
                this._readyPromise = null;
            }
        }
    }

    global.offlineOCR = new OfflineOCR();
})(window);
