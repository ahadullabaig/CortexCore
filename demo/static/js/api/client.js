/**
 * PHASE 4: Error Boundaries & Retry Logic
 *
 * Provides:
 * - HTTP status code handling
 * - Timeout management
 * - Exponential backoff retry
 * - User-friendly error messages
 * - Toast notifications
 */

/**
 * Error Handler with Retry Logic
 */
export const ErrorHandler = {
    // Configuration
    config: {
        defaultTimeout: 10000, // 10 seconds
        maxRetries: 3,
        baseRetryDelay: 1000, // 1 second
        retryableStatuses: [408, 429, 500, 502, 503, 504] // HTTP status codes to retry
    },

    /**
     * Enhanced fetch with timeout and retry logic
     */
    async fetchWithRetry(url, options = {}, retries = 0) {
        const timeout = options.timeout || this.config.defaultTimeout;

        try {
            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            // Add abort signal to options
            const fetchOptions = {
                ...options,
                signal: controller.signal
            };

            // Make request
            const response = await fetch(url, fetchOptions);
            clearTimeout(timeoutId);

            // Check HTTP status
            if (!response.ok) {
                return this._handleHTTPError(response, url, options, retries);
            }

            return response;

        } catch (error) {
            // Handle timeout
            if (error.name === 'AbortError') {
                return this._handleTimeout(url, options, retries);
            }

            // Handle network errors
            return this._handleNetworkError(error, url, options, retries);
        }
    },

    /**
     * Handle HTTP errors (4xx, 5xx)
     */
    async _handleHTTPError(response, url, options, retries) {
        const status = response.status;
        const statusText = response.statusText;

        console.error(`❌ HTTP ${status} error: ${url} - ${statusText}`);

        // Check if retryable
        if (this.config.retryableStatuses.includes(status) && retries < this.config.maxRetries) {
            console.warn(`🔄 Retrying request (${retries + 1}/${this.config.maxRetries})...`);
            await this._delay(this._calculateRetryDelay(retries));
            return this.fetchWithRetry(url, options, retries + 1);
        }

        // Not retryable or max retries reached
        const userMessage = this._getUserFriendlyMessage(status);
        this.showToast(userMessage, 'error');

        throw new Error(`HTTP ${status}: ${userMessage}`);
    },

    /**
     * Handle timeout errors
     */
    async _handleTimeout(url, options, retries) {
        console.error(`⏱️ Request timeout: ${url}`);

        // Retry with exponential backoff
        if (retries < this.config.maxRetries) {
            console.warn(`🔄 Retrying request after timeout (${retries + 1}/${this.config.maxRetries})...`);
            await this._delay(this._calculateRetryDelay(retries));
            return this.fetchWithRetry(url, options, retries + 1);
        }

        // Max retries reached
        this.showToast('Request timed out. Please check your connection and try again.', 'error');
        throw new Error('Request timeout');
    },

    /**
     * Handle network errors (connection failed, DNS failure, etc.)
     */
    async _handleNetworkError(error, url, options, retries) {
        console.error(`🌐 Network error: ${url}`, error);

        // Retry network errors
        if (retries < this.config.maxRetries) {
            console.warn(`🔄 Retrying request after network error (${retries + 1}/${this.config.maxRetries})...`);
            await this._delay(this._calculateRetryDelay(retries));
            return this.fetchWithRetry(url, options, retries + 1);
        }

        // Max retries reached
        this.showToast('Network error. Please check your connection.', 'error');
        throw error;
    },

    /**
     * Calculate retry delay with exponential backoff
     */
    _calculateRetryDelay(retries) {
        // Exponential backoff: 1s, 2s, 4s, 8s...
        return this.config.baseRetryDelay * Math.pow(2, retries);
    },

    /**
     * Delay helper for retry logic
     */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    /**
     * Get user-friendly error message from HTTP status
     */
    _getUserFriendlyMessage(status) {
        const messages = {
            400: 'Invalid request. Please check your input.',
            401: 'Authentication required. Please log in.',
            403: 'Access denied.',
            404: 'Resource not found. The server may be starting up.',
            408: 'Request timeout. Please try again.',
            429: 'Too many requests. Please wait a moment.',
            500: 'Server error. Please try again later.',
            502: 'Bad gateway. The server may be restarting.',
            503: 'Service unavailable. Please try again in a moment.',
            504: 'Gateway timeout. The server is taking too long to respond.'
        };

        return messages[status] || `Error ${status}. Please try again.`;
    },

    /**
     * Toast Notification System
     */
    showToast(message, type = 'info') {
        // Check if toast container exists
        let container = document.getElementById('toast-container');

        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container';
            document.body.appendChild(container);
        }

        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        // Add icon based on type
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        // Create elements safely to prevent XSS
        const iconSpan = document.createElement('span');
        iconSpan.className = 'toast-icon';
        iconSpan.textContent = icons[type] || icons.info;

        const messageSpan = document.createElement('span');
        messageSpan.className = 'toast-message';
        messageSpan.textContent = message; // Use textContent to prevent XSS

        const closeBtn = document.createElement('button');
        closeBtn.className = 'toast-close';
        closeBtn.setAttribute('aria-label', 'Close notification');
        closeBtn.innerHTML = '&times;'; // Safe: no user input

        toast.appendChild(iconSpan);
        toast.appendChild(messageSpan);
        toast.appendChild(closeBtn);

        // Add to container
        container.appendChild(toast);

        // Setup close button (already created above)
        closeBtn.addEventListener('click', () => {
            toast.classList.add('toast-removing');
            setTimeout(() => toast.remove(), 300);
        });

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('toast-removing');
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);

        // Animate in
        setTimeout(() => toast.classList.add('toast-visible'), 10);

        console.log(`📢 Toast: [${type}] ${message}`);
    }
};

// Initialize error handler
console.log('🛡️ Error handler initialized');
