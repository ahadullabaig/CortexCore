/**
 * CortexCore - Demo JavaScript
 * ==============================================
 *
 * Owner: CS4 / Deployment Engineer
 * Phase: Days 3-30
 */

// ============================================
// Global State
// ============================================

// Legacy global variables (will be migrated to AppState)
let currentSignal = null;
let currentSpikes = null;

// Animation Management State
const animationState = {
    particles: [],
    activeTimeouts: [],
    activeAnimationFrames: [],
    isInitialized: false
};

// Spike Array Buffer Configuration (memory leak prevention)
const SPIKE_BUFFER_CONFIG = {
    MAX_POINTS: 10000, // Maximum points to render (prevents unbounded growth)
    BATCH_SIZE: 40,    // Number of batches for progressive reveal
    ANIMATION_DURATION: 1500 // Animation duration in ms
};

// ============================================
// PHASE 4: Centralized State Management System
// ============================================

/**
 * Application State Management
 *
 * Provides:
 * - Centralized state storage
 * - Type validation
 * - Pub/sub pattern for state changes
 * - State history (for debugging/undo)
 * - Immutability enforcement
 */
const AppState = {
    // Internal state storage (private)
    _state: {
        // Model & Device
        model: {
            loaded: false,
            accuracy: null,
            architecture: null,
            device: 'unknown'
        },

        // Signal Data
        signal: {
            data: null,
            condition: null,
            samplingRate: 250,
            duration: 10
        },

        // Spike Encoding
        spikes: {
            data: null,
            numNeurons: 0,
            numSteps: 0,
            totalSpikes: 0,
            firingRate: 0,
            sparsity: 0
        },

        // Prediction Results
        prediction: {
            className: null,
            confidence: 0,
            probabilities: [],
            inferenceTime: 0,
            timestamp: null
        },

        // UI State
        ui: {
            isGenerating: false,
            isPredicting: false,
            resultsVisible: false,
            selectedCondition: 'normal'
        }
    },

    // Subscribers (for pub/sub pattern)
    _subscribers: {},

    // State history (for debugging, max 10 states)
    _history: [],
    _maxHistorySize: 10,

    /**
     * Get current state (immutable copy)
     */
    getState(path = null) {
        if (path) {
            // Get nested property (e.g., 'model.loaded')
            return this._getNestedProperty(this._state, path);
        }
        // Return deep copy of entire state
        return JSON.parse(JSON.stringify(this._state));
    },

    /**
     * Update state (with validation and pub/sub)
     */
    setState(path, value) {
        const oldValue = this._getNestedProperty(this._state, path);

        // Validate state change
        if (!this._validateStateChange(path, value)) {
            console.error(`❌ Invalid state change: ${path} = ${value}`);
            return false;
        }

        // Save to history
        this._addToHistory(path, oldValue, value);

        // Update state
        this._setNestedProperty(this._state, path, value);

        // Notify subscribers
        this._notifySubscribers(path, value, oldValue);

        console.log(`🔄 State updated: ${path} =`, value);
        return true;
    },

    /**
     * Subscribe to state changes
     * @param {string} path - State path to watch (e.g., 'model.loaded')
     * @param {function} callback - Function to call on change
     * @returns {function} Unsubscribe function
     */
    subscribe(path, callback) {
        if (!this._subscribers[path]) {
            this._subscribers[path] = [];
        }

        this._subscribers[path].push(callback);

        console.log(`👂 Subscribed to: ${path}`);

        // Return unsubscribe function
        return () => {
            this._subscribers[path] = this._subscribers[path].filter(cb => cb !== callback);
            console.log(`👋 Unsubscribed from: ${path}`);
        };
    },

    /**
     * Reset state to initial values
     */
    reset() {
        const initialState = {
            model: { loaded: false, accuracy: null, architecture: null, device: 'unknown' },
            signal: { data: null, condition: null, samplingRate: 250, duration: 10 },
            spikes: { data: null, numNeurons: 0, numSteps: 0, totalSpikes: 0, firingRate: 0, sparsity: 0 },
            prediction: { className: null, confidence: 0, probabilities: [], inferenceTime: 0, timestamp: null },
            ui: { isGenerating: false, isPredicting: false, resultsVisible: false, selectedCondition: 'normal' }
        };

        this._state = initialState;
        this._history = [];
        console.log('🔄 State reset to initial values');
    },

    /**
     * Get state history (for debugging)
     */
    getHistory() {
        return [...this._history];
    },

    // ===== Internal Helper Methods =====

    _getNestedProperty(obj, path) {
        return path.split('.').reduce((current, key) => current?.[key], obj);
    },

    _setNestedProperty(obj, path, value) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        const target = keys.reduce((current, key) => {
            if (!current[key]) current[key] = {};
            return current[key];
        }, obj);
        target[lastKey] = value;
    },

    _validateStateChange(path, value) {
        // Type validation based on path
        const validations = {
            'model.loaded': (v) => typeof v === 'boolean',
            'model.accuracy': (v) => v === null || typeof v === 'number',
            'ui.isGenerating': (v) => typeof v === 'boolean',
            'ui.isPredicting': (v) => typeof v === 'boolean',
            'signal.data': (v) => v === null || Array.isArray(v),
            'spikes.totalSpikes': (v) => typeof v === 'number' && v >= 0
        };

        const validator = validations[path];
        if (validator && !validator(value)) {
            return false;
        }

        return true;
    },

    _notifySubscribers(path, newValue, oldValue) {
        // Notify exact path subscribers
        if (this._subscribers[path]) {
            this._subscribers[path].forEach(callback => {
                try {
                    callback(newValue, oldValue);
                } catch (error) {
                    console.error(`❌ Subscriber error for ${path}:`, error);
                }
            });
        }

        // Notify wildcard subscribers (e.g., 'model.*' matches 'model.loaded')
        Object.keys(this._subscribers).forEach(subscribedPath => {
            if (subscribedPath.endsWith('.*') && path.startsWith(subscribedPath.slice(0, -2))) {
                this._subscribers[subscribedPath].forEach(callback => {
                    try {
                        callback(newValue, oldValue);
                    } catch (error) {
                        console.error(`❌ Wildcard subscriber error for ${subscribedPath}:`, error);
                    }
                });
            }
        });
    },

    _addToHistory(path, oldValue, newValue) {
        this._history.push({
            timestamp: Date.now(),
            path,
            oldValue,
            newValue
        });

        // Keep history size limited
        if (this._history.length > this._maxHistorySize) {
            this._history.shift();
        }
    }
};

// Initialize state management
console.log('🏗️ State management system initialized');

// ============================================
// PHASE 4: Error Boundaries & Retry Logic
// ============================================

/**
 * Error Handler with Retry Logic
 *
 * Provides:
 * - HTTP status code handling
 * - Timeout management
 * - Exponential backoff retry
 * - User-friendly error messages
 * - Toast notifications
 */
const ErrorHandler = {
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

        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-message">${message}</span>
            <button class="toast-close" aria-label="Close notification">&times;</button>
        `;

        // Add to container
        container.appendChild(toast);

        // Setup close button
        const closeBtn = toast.querySelector('.toast-close');
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

document.addEventListener('DOMContentLoaded', () => {
    console.log('🧠 CortexCore Demo initialized');

    // Phase 3: Initialize animation systems
    initializeAnimationSystems();

    // Check system health
    checkHealth();

    // Setup event listeners
    setupEventListeners();

    // Initialize empty plots
    initializePlots();

    // Setup cleanup on page unload
    window.addEventListener('beforeunload', cleanupAllAnimations);
});

// ============================================
// Accessibility Helper Functions
// ============================================

/**
 * Set button enabled/disabled state with ARIA synchronization
 * CRITICAL: Fixes WCAG 2.1 Level A violation (4.1.2 Name, Role, Value)
 *
 * @param {HTMLButtonElement} button - Button element to update
 * @param {boolean} isDisabled - Whether button should be disabled
 */
function setButtonState(button, isDisabled) {
    if (!button) {
        console.error('❌ setButtonState: button element is null');
        return;
    }

    button.disabled = isDisabled;
    button.setAttribute('aria-disabled', isDisabled.toString());

    // Log state change for debugging
    console.log(`🔘 Button "${button.id}" state: ${isDisabled ? 'disabled' : 'enabled'}`);
}

// ============================================
// Animation Cleanup Functions (Memory Leak Prevention)
// ============================================

/**
 * Clear all active timeouts and animation frames
 * Critical for preventing memory leaks
 */
function cleanupAllAnimations() {
    // Cancel all setTimeout calls
    animationState.activeTimeouts.forEach(id => clearTimeout(id));
    animationState.activeTimeouts = [];

    // Cancel all requestAnimationFrame calls
    animationState.activeAnimationFrames.forEach(id => cancelAnimationFrame(id));
    animationState.activeAnimationFrames = [];

    console.log('🧹 Cleaned up all active animations');
}

/**
 * Register timeout for cleanup tracking
 */
function managedSetTimeout(callback, delay) {
    const id = setTimeout(() => {
        callback();
        // Remove from tracking after execution
        const index = animationState.activeTimeouts.indexOf(id);
        if (index > -1) {
            animationState.activeTimeouts.splice(index, 1);
        }
    }, delay);

    animationState.activeTimeouts.push(id);
    return id;
}

/**
 * Register animation frame for cleanup tracking
 */
function managedRequestAnimationFrame(callback) {
    const id = requestAnimationFrame((time) => {
        callback(time);
        // Remove from tracking after execution
        const index = animationState.activeAnimationFrames.indexOf(id);
        if (index > -1) {
            animationState.activeAnimationFrames.splice(index, 1);
        }
    });

    animationState.activeAnimationFrames.push(id);
    return id;
}

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');
    const conditionSelect = document.getElementById('condition-select');

    generateBtn.addEventListener('click', generateSample);
    predictBtn.addEventListener('click', runPrediction);

    // Update condition preview on change
    conditionSelect.addEventListener('change', function(e) {
        const previews = {
            'normal': '70 BPM, Low Noise',
            'arrhythmia': '120 BPM, High Noise'
        };
        document.getElementById('condition-preview').textContent = previews[e.target.value];
    });
}

// ============================================
// API Calls
// ============================================

async function checkHealth() {
    try {
        // Use ErrorHandler for resilient fetching
        const response = await ErrorHandler.fetchWithRetry('/health', {
            timeout: 5000 // 5 second timeout for health checks
        });
        const data = await response.json();

        // Update state
        AppState.setState('model.loaded', data.model.loaded);
        AppState.setState('model.device', data.device);
        if (data.model.val_acc && data.model.val_acc !== 'N/A') {
            AppState.setState('model.accuracy', data.model.val_acc);
        }

        // Update status badges
        const modelStatus = document.getElementById('model-status');
        const deviceStatus = document.getElementById('device-status');

        if (data.model.loaded) {
            modelStatus.textContent = 'Loaded';
            modelStatus.className = 'status-badge success';

            // Update model accuracy if available
            const accElement = document.getElementById('model-accuracy');
            if (data.model.val_acc && data.model.val_acc !== 'N/A') {
                accElement.textContent = `${data.model.val_acc.toFixed(2)}%`;
            } else {
                accElement.textContent = 'N/A';
            }
        } else {
            modelStatus.textContent = 'Not Loaded';
            modelStatus.className = 'status-badge warning';
        }

        deviceStatus.textContent = data.device.toUpperCase();
        deviceStatus.className = data.device === 'cuda' ? 'status-badge success' : 'status-badge info';

        console.log('✅ Health check complete:', data);
    } catch (error) {
        console.error('❌ Health check failed:', error);
        const modelStatus = document.getElementById('model-status');
        modelStatus.textContent = 'Error';
        modelStatus.className = 'status-badge danger';

        // Don't show toast for health check failures (silent)
        // ErrorHandler already logged the error
    }
}

async function generateSample() {
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');
    const condition = document.getElementById('condition-select').value;

    try {
        setButtonState(generateBtn, true); // Disable with ARIA sync
        generateBtn.textContent = 'Generating...';

        // CRITICAL: Clean up any ongoing animations before new generation
        // This prevents memory leaks from accumulated animation frames
        cleanupAllAnimations();

        // Use ErrorHandler for resilient fetching with retry logic
        const response = await ErrorHandler.fetchWithRetry('/api/generate_sample', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                condition: condition,
                duration: 10,
                sampling_rate: 250
            }),
            timeout: 15000 // 15 second timeout for data generation
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        currentSignal = data.signal;

        // Plot ECG
        plotECG(data.signal, data.condition);

        // Generate and plot spikes
        await generateSpikes(data.signal);

        // Enable prediction
        setButtonState(predictBtn, false); // Enable with ARIA sync

        // Show success notification
        ErrorHandler.showToast('ECG signal generated successfully', 'success');

        console.log('✅ Sample generated:', data);

    } catch (error) {
        console.error('❌ Generation failed:', error);
        // ErrorHandler already shows toast notification for fetch errors
        // Only show toast for other types of errors
        if (!error.message.includes('HTTP')) {
            ErrorHandler.showToast(`Generation failed: ${error.message}`, 'error');
        }
    } finally {
        setButtonState(generateBtn, false); // Re-enable with ARIA sync
        generateBtn.textContent = 'Generate ECG Sample';
    }
}

async function generateSpikes(signal) {
    try {
        // Measure spike encoding time
        const data = await measureVisualizationTime('spike-encoding', async () => {
            // Use ErrorHandler for resilient fetching with retry logic
            const response = await ErrorHandler.fetchWithRetry('/api/visualize_spikes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    signal: signal
                }),
                timeout: 10000 // 10 second timeout for spike encoding
            });

            return await response.json();
        });

        if (data.error) {
            throw new Error(data.error);
        }

        currentSpikes = data;

        // Update spike statistics with animated counters
        const totalSpikes = data.spike_times.length;
        const duration = 10.0; // ECG signal duration is always 10 seconds
        const firingRate = totalSpikes / duration;
        const sparsity = (totalSpikes / (data.num_neurons * data.num_steps)) * 100;

        // Animate all three counters with staggered timing
        animateCountUp(document.getElementById('total-spikes'), 0, totalSpikes, 800);
        animateCountUp(document.getElementById('firing-rate'), 0, firingRate, 800, { suffix: ' Hz', decimals: 1 });
        animateCountUp(document.getElementById('sparsity'), 0, sparsity, 800, { suffix: '%', decimals: 1 });

        plotSpikes(data);

        console.log('✅ Spikes generated:', data);

    } catch (error) {
        console.error('❌ Spike generation failed:', error);
        // ErrorHandler already shows toast notification for fetch errors
        // Only show toast for other types of errors
        if (!error.message.includes('HTTP')) {
            ErrorHandler.showToast(`Spike generation failed: ${error.message}`, 'error');
        }
    }
}

async function runPrediction() {
    if (!currentSignal) {
        ErrorHandler.showToast('Please generate a sample first', 'warning');
        return;
    }

    const predictBtn = document.getElementById('predict-btn');

    try {
        setButtonState(predictBtn, true); // Disable with ARIA sync
        predictBtn.textContent = 'Predicting...';

        // Use ErrorHandler for resilient fetching with retry logic
        const response = await ErrorHandler.fetchWithRetry('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                signal: currentSignal,
                encode: true
            }),
            timeout: 20000 // 20 second timeout for neural network inference
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);

        // Show success notification
        ErrorHandler.showToast(`Classification: ${data.class_name} (${(data.confidence * 100).toFixed(1)}% confidence)`, 'success');

        console.log('✅ Prediction complete:', data);

    } catch (error) {
        console.error('❌ Prediction failed:', error);
        // ErrorHandler already shows toast notification for fetch errors
        // Only show toast for other types of errors
        if (!error.message.includes('HTTP')) {
            ErrorHandler.showToast(`Prediction failed: ${error.message}`, 'error');
        }
    } finally {
        setButtonState(predictBtn, false); // Re-enable with ARIA sync
        predictBtn.textContent = 'Run Prediction';
    }
}

// ============================================
// Visualization
// ============================================

// ============================================
// PHASE 4: Dynamic Plotly Theme System
// ============================================

/**
 * Plotly Theme Configuration - Reads from CSS Variables
 * This creates a centralized theme that stays in sync with CSS
 */
const PlotlyTheme = {
    // Cache for CSS variables (populated on first access)
    _cssVars: null,

    /**
     * Read CSS custom properties from document
     * Cached for performance
     */
    getCSSVariables() {
        if (!this._cssVars) {
            const root = document.documentElement;
            const computed = getComputedStyle(root);

            this._cssVars = {
                // Fonts
                fontPrimary: computed.getPropertyValue('--font-primary').trim() || 'JetBrains Mono, monospace',

                // Colors - Text
                textPrimary: computed.getPropertyValue('--text-primary').trim() || '#e4e7ed',
                textSecondary: computed.getPropertyValue('--text-secondary').trim() || '#b4bcd0',
                textTertiary: computed.getPropertyValue('--text-tertiary').trim() || '#8b92a0',

                // Colors - Accent
                neuralCyan: computed.getPropertyValue('--neural-cyan').trim() || '#00d9ff',
                neuralBlue: computed.getPropertyValue('--neural-blue').trim() || '#0088ff',
                neuralPurple: computed.getPropertyValue('--neural-purple').trim() || '#a855f7',
                neuralTeal: computed.getPropertyValue('--neural-teal').trim() || '#14b8a6',

                // Colors - Clinical
                clinicalNormal: computed.getPropertyValue('--clinical-normal').trim() || '#10b981',
                clinicalCritical: computed.getPropertyValue('--clinical-critical').trim() || '#ef4444',
                clinicalWarning: computed.getPropertyValue('--clinical-warning').trim() || '#f59e0b',
                clinicalInfo: computed.getPropertyValue('--clinical-info').trim() || '#3b82f6',

                // Backgrounds
                bgPrimary: computed.getPropertyValue('--bg-primary').trim() || '#0a0e1a',
                bgSecondary: computed.getPropertyValue('--bg-secondary').trim() || '#0f1419',

                // Borders
                borderSubtle: computed.getPropertyValue('--border-subtle').trim() || '#1f2937',
            };

            console.log('🎨 Plotly theme initialized with CSS variables:', this._cssVars);
        }

        return this._cssVars;
    },

    /**
     * Invalidate CSS variable cache
     * Call this if CSS theme changes dynamically
     */
    invalidateCache() {
        this._cssVars = null;
        console.log('🔄 Plotly theme cache invalidated');
    },

    /**
     * Generate base layout configuration
     * Applies to all Plotly charts for consistency
     */
    getBaseLayout(customTitle = '') {
        const vars = this.getCSSVariables();

        return {
            title: {
                text: customTitle,
                font: {
                    family: vars.fontPrimary,
                    color: vars.textSecondary,
                    size: 16
                }
            },
            plot_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent to show card background
            paper_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent
            margin: { t: 60, r: 20, b: 60, l: 70 },
            font: {
                family: vars.fontPrimary,
                color: vars.textSecondary,
                size: 12
            },
            hoverlabel: {
                bgcolor: vars.bgSecondary,
                bordercolor: vars.neuralCyan,
                font: {
                    family: vars.fontPrimary,
                    color: vars.textPrimary,
                    size: 12
                }
            },
            // Modebar (toolbar) styling
            modebar: {
                bgcolor: 'rgba(0, 0, 0, 0)',
                color: vars.textTertiary,
                activecolor: vars.neuralCyan
            }
        };
    },

    /**
     * Generate axis configuration
     */
    getAxisConfig(title = '', options = {}) {
        const vars = this.getCSSVariables();

        return {
            title: {
                text: title,
                font: {
                    family: vars.fontPrimary,
                    color: vars.textSecondary,
                    size: 13
                }
            },
            gridcolor: `rgba(180, 188, 208, 0.08)`, // Subtle grid
            zerolinecolor: `rgba(180, 188, 208, 0.15)`, // Slightly more visible zero line
            color: vars.textTertiary, // Tick labels
            tickfont: {
                family: vars.fontPrimary,
                size: 11
            },
            ...options // Allow overrides
        };
    },

    /**
     * Get color for condition/classification
     */
    getConditionColor(condition) {
        const vars = this.getCSSVariables();

        const colorMap = {
            'normal': vars.clinicalNormal,
            'arrhythmia': vars.clinicalCritical,
            'warning': vars.clinicalWarning,
            'info': vars.clinicalInfo
        };

        return colorMap[condition.toLowerCase()] || vars.neuralCyan;
    },

    /**
     * Get spike/neural activity color
     */
    getSpikeColor() {
        const vars = this.getCSSVariables();
        return vars.neuralPurple;
    },

    /**
     * Get Plotly configuration with accessible modebar
     * WCAG 2.1 AA: Ensures all chart controls have descriptive labels
     *
     * Plotly's default buttons already have accessible tooltips via title attributes.
     * We just need to:
     * 1. Remove confusing/duplicate buttons
     * 2. Enable the modebar
     * 3. Set locale for better i18n
     *
     * Default buttons (with built-in titles):
     * - toImage: "Download plot as png"
     * - zoom2d: "Zoom"
     * - pan2d: "Pan"
     * - zoomIn2d: "Zoom in"
     * - zoomOut2d: "Zoom out"
     * - autoScale2d: "Autoscale"
     * - resetScale2d: "Reset axes"
     *
     * @param {Object} options - Override options
     * @returns {Object} Plotly config object
     */
    getAccessibleConfig(options = {}) {
        return {
            responsive: true,
            displayModeBar: true,
            displaylogo: false, // Remove Plotly logo
            // Remove confusing selection tools, keep essential zoom/pan controls
            modeBarButtonsToRemove: [
                'lasso2d',      // Lasso select (not useful for time series)
                'select2d'      // Box select (not useful for time series)
            ],
            // Enable proper ARIA labels via locale
            locale: 'en',
            // Ensure tooltips are shown (default true, but explicit for clarity)
            displayModeBarTooltips: true,
            ...options // Allow overrides
        };
    }
};

/**
 * Wait for fonts to load before initializing Plotly
 * Prevents FOUT (Flash of Unstyled Text) in charts
 */
async function waitForFonts() {
    try {
        // Check if document.fonts API is available
        if ('fonts' in document) {
            await document.fonts.ready;
            console.log('✅ Fonts loaded, ready for Plotly rendering');
        } else {
            // Fallback: wait 100ms for fonts to load
            console.warn('⚠️ document.fonts API not available, using fallback delay');
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    } catch (error) {
        console.warn('⚠️ Font loading check failed:', error);
    }
}

function initializePlots() {
    // Wait for fonts before rendering
    waitForFonts().then(() => {
        // Get base theme configuration
        const baseLayout = PlotlyTheme.getBaseLayout('No signal generated yet');

        // Initialize empty ECG plot with dynamic theme
        const ecgLayout = {
            ...baseLayout,
            title: {
                ...baseLayout.title,
                text: 'No signal generated yet'
            },
            xaxis: PlotlyTheme.getAxisConfig('Sample'),
            yaxis: PlotlyTheme.getAxisConfig('Amplitude')
        };

        // Use accessible config for better WCAG compliance
        Plotly.newPlot('ecg-plot', [], ecgLayout, PlotlyTheme.getAccessibleConfig());

        // Initialize empty spike plot with dynamic theme
        const spikeLayout = {
            ...baseLayout,
            title: {
                ...baseLayout.title,
                text: 'No spikes generated yet'
            },
            xaxis: PlotlyTheme.getAxisConfig('Time Step'),
            yaxis: PlotlyTheme.getAxisConfig('Neuron Index')
        };

        // Use accessible config for better WCAG compliance
        Plotly.newPlot('spike-plot', [], spikeLayout, PlotlyTheme.getAccessibleConfig());

        console.log('📊 Plotly charts initialized with dynamic CSS theme');
    });
}

function plotECG(signal, condition) {
    // Use PlotlyTheme to get dynamic colors from CSS variables
    const conditionColor = PlotlyTheme.getConditionColor(condition);

    const trace = {
        y: signal,
        type: 'scatter',
        mode: 'lines',
        name: condition.charAt(0).toUpperCase() + condition.slice(1),
        line: {
            color: conditionColor,
            width: 2.5,
            shape: 'spline', // Smooth curve for more organic look
            smoothing: 0.8
        },
        hovertemplate: '<b>Sample</b>: %{x}<br><b>Amplitude</b>: %{y:.3f}<extra></extra>'
    };

    // Get base layout with dynamic theme
    const baseLayout = PlotlyTheme.getBaseLayout(
        `ECG Signal - ${condition.charAt(0).toUpperCase() + condition.slice(1)}`
    );

    const layout = {
        ...baseLayout,
        xaxis: PlotlyTheme.getAxisConfig('Sample'),
        yaxis: PlotlyTheme.getAxisConfig('Amplitude (normalized)'),
        showlegend: false // Single trace, no need for legend
    };

    // Use accessible config with descriptive button labels
    Plotly.newPlot('ecg-plot', [trace], layout, PlotlyTheme.getAccessibleConfig());

    // Add progressive trace drawing animation for visual impact
    // This simulates the ECG being "drawn" in real-time
    // PERFORMANCE FIX: Use Plotly.extendTraces for better performance
    const totalPoints = signal.length;
    const animationDuration = 2000; // 2 seconds total
    const numBatches = 60; // 60 updates for smooth 30fps animation
    const pointsPerBatch = Math.ceil(totalPoints / numBatches);
    const delayPerBatch = animationDuration / numBatches;

    let currentBatch = 0;

    // Initialize with empty data
    Plotly.restyle('ecg-plot', {
        y: [[]]
    }, [0]);

    function drawNextSegment() {
        if (currentBatch >= numBatches) {
            // Final update with complete data
            Plotly.restyle('ecg-plot', {
                y: [signal]
            }, [0]);
            return;
        }

        const startIdx = currentBatch * pointsPerBatch;
        const endIdx = Math.min(startIdx + pointsPerBatch, totalPoints);
        const batchData = signal.slice(startIdx, endIdx);

        // Use extendTraces for better performance
        Plotly.extendTraces('ecg-plot', {
            y: [batchData]
        }, [0]);

        currentBatch++;

        if (currentBatch < numBatches) {
            // Use managed timeout for cleanup tracking
            managedSetTimeout(drawNextSegment, delayPerBatch);
        }
    }

    // Start drawing animation after brief delay
    managedSetTimeout(drawNextSegment, 100);
}

function plotSpikes(spikeData) {
    // MEMORY LEAK FIX: Implement data subsampling if spike count exceeds buffer limit
    const totalSpikes = spikeData.spike_times.length;
    let displayData = { spike_times: spikeData.spike_times, neuron_ids: spikeData.neuron_ids };

    if (totalSpikes > SPIKE_BUFFER_CONFIG.MAX_POINTS) {
        console.warn(`⚠️ Spike count (${totalSpikes}) exceeds buffer limit (${SPIKE_BUFFER_CONFIG.MAX_POINTS}). Subsampling for performance.`);

        // Subsample uniformly to stay within buffer limit
        const step = Math.ceil(totalSpikes / SPIKE_BUFFER_CONFIG.MAX_POINTS);
        displayData = {
            spike_times: spikeData.spike_times.filter((_, idx) => idx % step === 0),
            neuron_ids: spikeData.neuron_ids.filter((_, idx) => idx % step === 0)
        };
    }

    // Use PlotlyTheme for consistent spike color
    const spikeColor = PlotlyTheme.getSpikeColor();

    const trace = {
        x: displayData.spike_times,
        y: displayData.neuron_ids,
        mode: 'markers',
        type: 'scatter',
        name: 'Neural Spikes',
        marker: {
            color: spikeColor,
            size: 5,
            symbol: 'line-ns-open',
            line: {
                width: 2,
                color: spikeColor
            },
            opacity: 0.9
        },
        hovertemplate: '<b>Time</b>: %{x}<br><b>Neuron</b>: %{y}<extra></extra>'
    };

    // Get base layout with dynamic theme
    const baseLayout = PlotlyTheme.getBaseLayout(
        `Spike Raster Plot - ${totalSpikes.toLocaleString()} spikes${totalSpikes > SPIKE_BUFFER_CONFIG.MAX_POINTS ? ' (subsampled)' : ''}`
    );

    const layout = {
        ...baseLayout,
        xaxis: PlotlyTheme.getAxisConfig('Time Step', {
            range: [0, spikeData.num_steps]
        }),
        yaxis: PlotlyTheme.getAxisConfig('Neuron Index', {
            range: [-1, spikeData.num_neurons]
        }),
        showlegend: false // Single trace, no need for legend
    };

    // Use accessible config with descriptive button labels
    Plotly.newPlot('spike-plot', [trace], layout, PlotlyTheme.getAccessibleConfig());

    // Add progressive reveal animation for temporal dynamics visualization
    // Group spikes by time step
    const spikesByTime = {};
    for (let i = 0; i < displayData.spike_times.length; i++) {
        const timeStep = displayData.spike_times[i];
        if (!spikesByTime[timeStep]) {
            spikesByTime[timeStep] = { times: [], neurons: [] };
        }
        spikesByTime[timeStep].times.push(timeStep);
        spikesByTime[timeStep].neurons.push(displayData.neuron_ids[i]);
    }

    const timeSteps = Object.keys(spikesByTime).map(Number).sort((a, b) => a - b);

    // Progressive reveal: gradually show spikes based on their temporal order
    // MEMORY LEAK FIX: Use Plotly.extendTraces instead of accumulating arrays
    const numBatches = Math.min(timeSteps.length, SPIKE_BUFFER_CONFIG.BATCH_SIZE);
    const batchSize = Math.ceil(timeSteps.length / numBatches);
    const delayPerBatch = SPIKE_BUFFER_CONFIG.ANIMATION_DURATION / numBatches;

    let currentBatch = 0;

    // Initialize with empty data for progressive reveal
    Plotly.restyle('spike-plot', {
        x: [[]],
        y: [[]]
    }, [0]);

    function revealNextBatch() {
        if (currentBatch >= numBatches) return;

        // Calculate batch data without accumulating in closure
        const startIdx = currentBatch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, timeSteps.length);

        const batchX = [];
        const batchY = [];

        for (let i = startIdx; i < endIdx; i++) {
            const timeStep = timeSteps[i];
            const spikes = spikesByTime[timeStep];
            batchX.push(...spikes.times);
            batchY.push(...spikes.neurons);
        }

        // Use extendTraces instead of restyle for better performance
        // This appends data instead of replacing entire trace
        Plotly.extendTraces('spike-plot', {
            x: [batchX],
            y: [batchY]
        }, [0]);

        currentBatch++;

        if (currentBatch < numBatches) {
            // Use managed timeout for cleanup tracking
            managedSetTimeout(revealNextBatch, delayPerBatch);
        }
    }

    // Start progressive reveal after brief delay
    managedSetTimeout(revealNextBatch, 200);
}

// ============================================
// Results Display
// ============================================

function displayResults(results) {
    // Show results card
    const resultsCard = document.getElementById('results-card');
    resultsCard.style.display = 'block';

    // Update classification
    const predictionClass = document.getElementById('prediction-class');
    predictionClass.textContent = results.class_name;
    predictionClass.style.color = results.class_name === 'Normal' ? '#4caf50' : '#f44336';

    // Update confidence
    const confidence = document.getElementById('prediction-confidence');
    confidence.textContent = `${(results.confidence * 100).toFixed(1)}%`;

    // Update inference time
    const inferenceTime = document.getElementById('inference-time');
    inferenceTime.textContent = `${results.inference_time_ms.toFixed(2)} ms`;

    // Update probability bars WITH ARIA ATTRIBUTES (accessibility)
    if (results.probabilities && results.probabilities.length >= 2) {
        const probNormal = results.probabilities[0] * 100;
        const probArrhythmia = results.probabilities[1] * 100;

        // Update normal probability bar
        const normalBar = document.getElementById('prob-normal');
        const normalTrack = normalBar.parentElement;
        normalBar.style.width = `${probNormal}%`;
        normalTrack.setAttribute('aria-valuenow', Math.round(probNormal));
        document.getElementById('prob-normal-text').textContent = `${probNormal.toFixed(1)}%`;

        // Update arrhythmia probability bar
        const arrhythmiaBar = document.getElementById('prob-arrhythmia');
        const arrhythmiaTrack = arrhythmiaBar.parentElement;
        arrhythmiaBar.style.width = `${probArrhythmia}%`;
        arrhythmiaTrack.setAttribute('aria-valuenow', Math.round(probArrhythmia));
        document.getElementById('prob-arrhythmia-text').textContent = `${probArrhythmia.toFixed(1)}%`;
    }

    // Show warning if using mock data
    if (results.warning) {
        console.warn('⚠️', results.warning);
    }

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Announce to screen readers (accessibility)
    announceToScreenReader(
        `Classification complete: ${results.class_name} with ${(results.confidence * 100).toFixed(1)}% confidence`
    );
}

/**
 * Announce message to screen readers using ARIA live region
 * Phase 4: Accessibility enhancement
 */
function announceToScreenReader(message) {
    // Create or get announcement container
    let announcer = document.getElementById('aria-announcer');

    if (!announcer) {
        announcer = document.createElement('div');
        announcer.id = 'aria-announcer';
        announcer.className = 'sr-only'; // Screen reader only (visually hidden)
        announcer.setAttribute('role', 'status');
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        document.body.appendChild(announcer);
    }

    // Clear previous announcement
    announcer.textContent = '';

    // Add new announcement after brief delay (ensures screen reader picks it up)
    setTimeout(() => {
        announcer.textContent = message;
    }, 100);
}

// ============================================
// Utility Functions
// ============================================

function formatTime(ms) {
    if (ms < 1) {
        return `${(ms * 1000).toFixed(2)} µs`;
    } else if (ms < 1000) {
        return `${ms.toFixed(2)} ms`;
    } else {
        return `${(ms / 1000).toFixed(2)} s`;
    }
}

// ============================================
// Console Styling
// ============================================

console.log('%c🧠 CortexCore Demo', 'font-size: 20px; font-weight: bold; color: #667eea;');
console.log('%cBrain-inspired computing for medical diagnosis', 'font-size: 14px; color: #666;');
console.log('');
console.log('📊 Features:');
console.log('  • Real-time ECG generation');
console.log('  • Spike-based encoding');
console.log('  • Energy-efficient inference');
console.log('  • Clinical-grade predictions');
console.log('');


// ============================================
// PHASE 3: ANIMATION SYSTEMS
// ============================================

// ============================================
// Animation System Initialization
// ============================================

function initializeAnimationSystems() {
    console.log('🎬 Initializing Phase 3 Animation Systems...');

    // 1. Detect device capabilities and set optimization class
    detectDeviceCapabilities();

    // 2. Initialize Intersection Observer for scroll-triggered animations
    initializeScrollAnimations();

    // 3. Setup button ripple effects
    setupButtonRipples();

    // 4. Create floating particles (if not low-end device)
    // MEMORY OPTIMIZATION: Reduced from 25 to 15 particles
    if (!document.body.classList.contains('low-end-device')) {
        createFloatingParticles(15); // 15 particles (reduced for performance)
    }

    // 5. Create grid scanner
    if (!document.body.classList.contains('low-end-device')) {
        createGridScanner();
    }

    // 6. Setup performance monitoring (dev mode)
    if (window.location.search.includes('debug')) {
        setupPerformanceMonitoring();
    }

    // 7. Pause animations when tab is inactive (battery saving)
    setupVisibilityHandler();

    console.log('✅ Animation systems initialized');
}


// ============================================
// Device Capability Detection
// ============================================

function detectDeviceCapabilities() {
    const body = document.body;

    // Check if low-end device (limited memory, slow CPU, or mobile)
    const isLowEnd = (
        navigator.hardwareConcurrency <= 2 || // 2 or fewer CPU cores
        navigator.deviceMemory <= 4 || // 4GB or less RAM
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    );

    // Check battery status if available
    if (navigator.getBattery) {
        navigator.getBattery().then(battery => {
            // If battery is low (<20%) and not charging, disable heavy animations
            if (battery.level < 0.2 && !battery.charging) {
                body.classList.add('low-end-device');
                console.log('⚡ Low battery detected - disabling heavy animations');
            }
        });
    }

    if (isLowEnd) {
        body.classList.add('low-end-device');
        console.log('📱 Low-end device detected - optimizing animations');
    } else {
        console.log('💪 High-performance device detected - full animations enabled');
    }
}


// ============================================
// Scroll-Triggered Animations (Intersection Observer)
// ============================================

function initializeScrollAnimations() {
    // Create intersection observer
    const observerOptions = {
        root: null, // viewport
        rootMargin: '0px',
        threshold: 0.1 // Trigger when 10% of element is visible
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add 'animated' class to trigger animation
                entry.target.classList.add('animated');

                // Optional: Unobserve after animation (one-time animation)
                // observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all elements with .animate-on-scroll class
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach(el => observer.observe(el));

    console.log(`👁️ Observing ${animatedElements.length} elements for scroll animations`);
}


// ============================================
// Button Ripple Effects
// ============================================

function setupButtonRipples() {
    const buttons = document.querySelectorAll('.btn');

    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Don't add ripple if button is disabled
            if (this.disabled) return;

            // Create ripple element
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');

            // Get click position relative to button
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Position ripple at click location
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';

            // Add ripple to button
            this.appendChild(ripple);

            // Remove ripple after animation completes (600ms)
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    console.log(`💫 Button ripple effects enabled for ${buttons.length} buttons`);
}


// ============================================
// Floating Particles System
// ============================================

/**
 * Create floating particles with proper lifecycle management
 * MEMORY LEAK FIX: Store particle references for cleanup
 */
function createFloatingParticles(count) {
    // Create container if it doesn't exist
    let container = document.querySelector('.neural-particles');

    if (!container) {
        container = document.createElement('div');
        container.classList.add('neural-particles');
        document.body.appendChild(container);
    }

    // Clear existing particles and their references
    clearFloatingParticles();

    // Create particles
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');

        // Randomize size
        const sizeRandom = Math.random();
        if (sizeRandom < 0.3) {
            particle.classList.add('small');
        } else if (sizeRandom > 0.7) {
            particle.classList.add('large');
        }

        // Random starting position
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';

        // Random animation duration (10-30 seconds)
        particle.style.animationDuration = (10 + Math.random() * 20) + 's';

        // Random animation delay for stagger effect
        particle.style.animationDelay = (Math.random() * 5) + 's';

        container.appendChild(particle);

        // Store reference for cleanup
        animationState.particles.push(particle);
    }

    console.log(`✨ Created ${count} floating particles (optimized for performance)`);
}

/**
 * Clear all floating particles and free memory
 */
function clearFloatingParticles() {
    const container = document.querySelector('.neural-particles');

    if (container) {
        container.innerHTML = '';
    }

    // Clear particle references
    animationState.particles = [];
}

/**
 * Pause particle animations (called when tab is hidden)
 */
function pauseParticleAnimations() {
    animationState.particles.forEach(particle => {
        particle.style.animationPlayState = 'paused';
    });
}

/**
 * Resume particle animations (called when tab is visible)
 */
function resumeParticleAnimations() {
    animationState.particles.forEach(particle => {
        particle.style.animationPlayState = 'running';
    });
}


// ============================================
// Grid Scanner Effect
// ============================================

function createGridScanner() {
    // Create scanner if it doesn't exist
    if (!document.querySelector('.grid-scanner')) {
        const scanner = document.createElement('div');
        scanner.classList.add('grid-scanner');
        document.body.appendChild(scanner);
        console.log('📡 Grid scanner effect enabled');
    }
}


// ============================================
// Number Counter Animations
// ============================================

function animateCountUp(element, start, end, duration = 1000, options = {}) {
    const { suffix = '', decimals = 0 } = options;
    const startTime = performance.now();
    const range = end - start;

    function updateCount(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out cubic)
        const easeProgress = 1 - Math.pow(1 - progress, 3);

        const current = start + range * easeProgress;
        const formattedValue = decimals > 0 ? current.toFixed(decimals) : Math.floor(current).toLocaleString();
        element.textContent = formattedValue + suffix;

        if (progress < 1) {
            requestAnimationFrame(updateCount);
        } else {
            const finalValue = decimals > 0 ? end.toFixed(decimals) : end.toLocaleString();
            element.textContent = finalValue + suffix;
            element.classList.add('counter-animated');
        }
    }

    requestAnimationFrame(updateCount);
}


// ============================================
// Progress Bar Smooth Animation
// ============================================

function animateProgressBar(element, targetWidth, duration = 800) {
    const startTime = performance.now();
    const startWidth = parseFloat(element.style.width) || 0;
    const widthChange = targetWidth - startWidth;

    function updateWidth(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeProgress = 1 - Math.pow(1 - progress, 2);

        const currentWidth = startWidth + widthChange * easeProgress;
        element.style.width = currentWidth + '%';

        if (progress < 1) {
            requestAnimationFrame(updateWidth);
        }
    }

    requestAnimationFrame(updateWidth);
}


// ============================================
// Loading Spinner Management
// ============================================

function showLoadingSpinner(container, size = 'normal') {
    const spinner = document.createElement('div');
    spinner.classList.add('loading-spinner');
    if (size === 'small') {
        spinner.classList.add('small');
    }
    spinner.id = 'active-spinner';

    if (typeof container === 'string') {
        document.getElementById(container).appendChild(spinner);
    } else {
        container.appendChild(spinner);
    }

    return spinner;
}

function hideLoadingSpinner() {
    const spinner = document.getElementById('active-spinner');
    if (spinner) {
        spinner.remove();
    }
}


// ============================================
// Performance Monitoring (Debug Mode)
// ============================================

function setupPerformanceMonitoring() {
    let lastTime = performance.now();
    let frames = 0;
    let fps = 60;

    // Create FPS counter display
    const fpsDisplay = document.createElement('div');
    fpsDisplay.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.8);
        color: #00d9ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        padding: 8px 12px;
        border-radius: 4px;
        border: 1px solid #00d9ff;
        z-index: 10000;
        pointer-events: none;
    `;
    fpsDisplay.id = 'fps-counter';
    document.body.appendChild(fpsDisplay);

    function updateFPS() {
        const currentTime = performance.now();
        frames++;

        if (currentTime >= lastTime + 1000) {
            fps = Math.round((frames * 1000) / (currentTime - lastTime));
            frames = 0;
            lastTime = currentTime;

            // Update display with FPS and memory
            const memoryInfo = performance.memory ?
                ` | Heap: ${(performance.memory.usedJSHeapSize / 1048576).toFixed(1)}MB` :
                '';

            // Include visualization metrics if available
            const vizMetrics = (performanceMetrics.ecgRenderTime > 0 || performanceMetrics.spikeRenderTime > 0) ?
                `\nECG: ${performanceMetrics.ecgRenderTime.toFixed(1)}ms | Spikes: ${performanceMetrics.spikeRenderTime.toFixed(1)}ms | Encoding: ${performanceMetrics.spikeEncodingTime.toFixed(1)}ms` :
                '';

            fpsDisplay.textContent = `FPS: ${fps}${memoryInfo}${vizMetrics}`;
            fpsDisplay.style.borderColor = fps < 30 ? '#ef4444' : fps < 50 ? '#f59e0b' : '#00d9ff';
        }

        requestAnimationFrame(updateFPS);
    }

    requestAnimationFrame(updateFPS);
    console.log('📊 Performance monitoring enabled (debug mode)');
}

// Global performance metrics storage
const performanceMetrics = {
    ecgRenderTime: 0,
    spikeRenderTime: 0,
    spikeEncodingTime: 0,
    lastUpdate: Date.now()
};

// Helper function to measure and display visualization performance
function measureVisualizationTime(name, callback) {
    const startMark = `${name}-start`;
    const endMark = `${name}-end`;
    const measureName = `${name}-duration`;

    performance.mark(startMark);

    // Execute the callback (can be sync or async)
    const result = callback();

    // Handle both sync and async callbacks
    if (result && typeof result.then === 'function') {
        return result.then((value) => {
            performance.mark(endMark);
            const measure = performance.measure(measureName, startMark, endMark);
            updatePerformanceMetrics(name, measure.duration);
            return value;
        });
    } else {
        performance.mark(endMark);
        const measure = performance.measure(measureName, startMark, endMark);
        updatePerformanceMetrics(name, measure.duration);
        return result;
    }
}

// Update stored metrics (display updated by FPS loop in debug mode)
function updatePerformanceMetrics(name, duration) {
    if (name.includes('ecg')) {
        performanceMetrics.ecgRenderTime = duration;
    } else if (name.includes('spike-encoding')) {
        performanceMetrics.spikeEncodingTime = duration;
    } else if (name.includes('spike')) {
        performanceMetrics.spikeRenderTime = duration;
    }
    performanceMetrics.lastUpdate = Date.now();

    console.log(`⏱️  ${name}: ${duration.toFixed(2)}ms`);
}


// ============================================
// Visibility Handler (Pause animations when tab inactive)
// ============================================

/**
 * Pause heavy animations when tab is hidden (battery saving)
 * MEMORY LEAK FIX: Target specific animations, not all children
 */
function setupVisibilityHandler() {
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            // Pause particle animations specifically
            pauseParticleAnimations();

            // Pause grid scanner
            const gridScanner = document.querySelector('.grid-scanner');
            if (gridScanner) {
                gridScanner.style.animationPlayState = 'paused';
            }

            console.log('⏸️ Heavy animations paused (tab inactive - saving battery)');
        } else {
            // Resume particle animations
            resumeParticleAnimations();

            // Resume grid scanner
            const gridScanner = document.querySelector('.grid-scanner');
            if (gridScanner) {
                gridScanner.style.animationPlayState = 'running';
            }

            console.log('▶️ Animations resumed (tab active)');
        }
    });
}


// ============================================
// Enhanced Plotly Animations
// ============================================

// Override plot functions to add custom animations
const originalPlotECG = plotECG;
const originalPlotSpikes = plotSpikes;

// Enhanced ECG plotting with animation
window.plotECG = function(signal, condition) {
    // Measure render time and call original function
    measureVisualizationTime('ecg-render', () => {
        originalPlotECG(signal, condition);
    });

    // Add entrance animation to plot container
    const plotContainer = document.getElementById('ecg-plot');
    if (plotContainer) {
        plotContainer.style.animation = 'fadeInUp 0.6s ease-out';
    }
};

// Enhanced spike plotting with animation
window.plotSpikes = function(spikeData) {
    // Measure render time and call original function
    measureVisualizationTime('spike-render', () => {
        originalPlotSpikes(spikeData);
    });

    // Add entrance animation to plot container
    const plotContainer = document.getElementById('spike-plot');
    if (plotContainer) {
        plotContainer.style.animation = 'scaleIn 0.6s ease-out';
    }

    // Animate spike statistics with counter effect (now handled in generateSpikes)
    // Counter animations are already applied in generateSpikes function
};
