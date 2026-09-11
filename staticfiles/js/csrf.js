// core/static/core/js/csrf.js

/**
 * Get CSRF token from Django's cookie
 */
function getCSRFToken() {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, 'csrftoken'.length + 1) === ('csrftoken=')) {
                cookieValue = decodeURIComponent(cookie.substring('csrftoken'.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

/**
 * Create headers with CSRF token for fetch requests
 */
function getCSRFHeaders() {
    return {
        'X-CSRFToken': getCSRFToken(),
        'Content-Type': 'application/json',
    };
}

/**
 * Wrapper for fetch with CSRF token
 */
async function fetchWithCSRF(url, options = {}) {
    const defaultOptions = {
        method: 'POST',
        headers: getCSRFHeaders(),
        credentials: 'same-origin',
    };
    
    // Merge options
    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...(options.headers || {}),
        },
    };
    
    return fetch(url, mergedOptions);
}