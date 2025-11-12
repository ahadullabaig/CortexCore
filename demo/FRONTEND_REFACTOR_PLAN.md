# Frontend Refactoring Plan: CortexCore Demo

**Version**: 1.0
**Date**: 2025-11-12
**Status**: Planning Phase
**Estimated Total Time**: 6-8 hours
**Risk Level**: LOW (purely organizational, easily reversible)

---

## Executive Summary

### Current State
- `static/script.js`: 2,076 lines (state, API, animations, visualizations, events)
- `static/style.css`: 2,000+ lines (27+ component sections, animations, utilities)
- **Pain Points**: Merge conflicts, difficult navigation, cache invalidation, parallel development blocked

### Target State
- Modular JavaScript (ES6 modules): 20+ focused files
- Modular CSS (@import): 25+ component files
- **Benefits**: Parallel development, better caching, easier testing, cleaner git diffs

### Success Criteria
- ✅ Zero breaking changes to functionality
- ✅ All tests pass (health check, generation, prediction, visualization, animations)
- ✅ Improved developer experience (file navigation, merge conflicts)
- ✅ Instant rollback capability (preserve original files)

---

## Risk Assessment

### ✅ LOW RISK Factors
1. **No Logic Changes** - Pure code movement between files
2. **Flask Agnostic** - Static file structure doesn't affect Flask routing
3. **Native ES6 Modules** - No build step required, all modern browsers support
4. **Incremental Approach** - Test after each phase, rollback if issues
5. **Backups Preserved** - Original files remain until full verification

### ⚠️ Known Gotchas & Mitigations

| Gotcha | Impact | Mitigation |
|--------|--------|------------|
| Global variables (`currentSignal`, `currentSpikes`) | Module isolation issues | Export from modules, import where needed |
| HTML inline handlers (`onclick="..."`) | Functions not in global scope | Temporarily expose on `window` object |
| Module loading order | Dependency resolution errors | main.js orchestrates all imports |
| CSS specificity changes | Visual regressions | Maintain exact selector order in imports |
| Browser caching during dev | Stale module references | Add `?v=timestamp` to script tags temporarily |

---

## Prerequisites

### Before Starting
- [ ] Create feature branch: `git checkout -b refactor/frontend-modularization`
- [ ] Backup current state: `cp -r demo/static demo/static.backup`
- [ ] Verify demo runs: `make demo` → http://localhost:5000
- [ ] Document current behavior: Record video of full user flow
- [ ] Ensure clean working directory: `git status` (no uncommitted changes)

### Development Environment
- [ ] Modern browser with ES6 module support (Chrome 61+, Firefox 60+, Safari 11+)
- [ ] Browser DevTools open for console error monitoring
- [ ] Flask debug mode enabled: `FLASK_DEBUG=1` in `.env`

---

## Phase 1: Foundation - Utilities & Configuration

**Goal**: Extract zero-dependency utility functions and configuration constants
**Time Estimate**: 1.5 hours
**Risk**: LOWEST (no dependencies on other modules)

### Files to Create

```
demo/static/
├── js/
│   ├── utils/
│   │   ├── device.js          # detectDeviceCapabilities()
│   │   ├── performance.js     # performanceMetrics, measureVisualizationTime()
│   │   ├── timing.js          # formatTime(), managedSetTimeout(), managedRequestAnimationFrame()
│   │   └── accessibility.js   # setButtonState(), announceToScreenReader()
│   └── core/
│       └── config.js          # SPIKE_BUFFER_CONFIG, animationState
```

### Step-by-Step Instructions

#### 1.1 Create Directory Structure
```bash
cd demo/static
mkdir -p js/utils js/core
```

#### 1.2 Extract `js/utils/device.js`

**Lines to Extract**: script.js:1601-1628

```javascript
/**
 * Device Capability Detection
 * Detects low-end devices and optimizes animations accordingly
 */

/**
 * Detect device capabilities and apply optimization class
 * Sets .low-end-device class on body if device has limited resources
 */
export function detectDeviceCapabilities() {
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
```

#### 1.3 Extract `js/utils/performance.js`

**Lines to Extract**: script.js:1951-1998

```javascript
/**
 * Performance Monitoring Utilities
 * Tracks render times, memory usage, and frame rates
 */

// Global performance metrics storage
export const performanceMetrics = {
    ecgRenderTime: 0,
    spikeRenderTime: 0,
    spikeEncodingTime: 0,
    lastUpdate: Date.now()
};

/**
 * Measure and display visualization performance
 * @param {string} name - Metric name (e.g., 'ecg-render', 'spike-encoding')
 * @param {function} callback - Function to measure (sync or async)
 * @returns {*} Result of callback
 */
export function measureVisualizationTime(name, callback) {
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

/**
 * Update stored metrics (display updated by FPS loop in debug mode)
 * @private
 */
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
```

#### 1.4 Extract `js/utils/timing.js`

**Lines to Extract**: script.js:1529-1537, 544-573

```javascript
/**
 * Timing Utilities
 * Time formatting and managed timeout/animation frame tracking
 */

/**
 * Format milliseconds to human-readable string
 * @param {number} ms - Time in milliseconds
 * @returns {string} Formatted time (µs, ms, or s)
 */
export function formatTime(ms) {
    if (ms < 1) {
        return `${(ms * 1000).toFixed(2)} µs`;
    } else if (ms < 1000) {
        return `${ms.toFixed(2)} ms`;
    } else {
        return `${(ms / 1000).toFixed(2)} s`;
    }
}

/**
 * Register timeout for cleanup tracking
 * Prevents memory leaks by tracking active timeouts
 * @param {function} callback - Function to call after delay
 * @param {number} delay - Delay in milliseconds
 * @returns {number} Timeout ID
 */
export function managedSetTimeout(callback, delay) {
    // Import animationState dynamically to avoid circular dependency
    const { animationState } = await import('../core/config.js');

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
 * Prevents memory leaks by tracking active animation frames
 * @param {function} callback - Function to call on next frame
 * @returns {number} Animation frame ID
 */
export function managedRequestAnimationFrame(callback) {
    // Import animationState dynamically to avoid circular dependency
    const { animationState } = await import('../core/config.js');

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
```

#### 1.5 Extract `js/utils/accessibility.js`

**Lines to Extract**: script.js:498-519, 1499-1523

```javascript
/**
 * Accessibility Helper Functions
 * WCAG 2.1 Level A/AA compliance utilities
 */

/**
 * Set button enabled/disabled state with ARIA synchronization
 * CRITICAL: Fixes WCAG 2.1 Level A violation (4.1.2 Name, Role, Value)
 *
 * @param {HTMLButtonElement} button - Button element to update
 * @param {boolean} isDisabled - Whether button should be disabled
 */
export function setButtonState(button, isDisabled) {
    if (!button) {
        console.error('❌ setButtonState: button element is null');
        return;
    }

    button.disabled = isDisabled;
    button.setAttribute('aria-disabled', isDisabled.toString());

    // Log state change for debugging
    console.log(`🔘 Button "${button.id}" state: ${isDisabled ? 'disabled' : 'enabled'}`);
}

/**
 * Announce message to screen readers using ARIA live region
 * Phase 4: Accessibility enhancement
 * @param {string} message - Message to announce
 */
export function announceToScreenReader(message) {
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
```

#### 1.6 Extract `js/core/config.js`

**Lines to Extract**: script.js:14-30

```javascript
/**
 * Core Configuration
 * Application-wide constants and state containers
 */

// Animation Management State
export const animationState = {
    particles: [],
    activeTimeouts: [],
    activeAnimationFrames: [],
    isInitialized: false
};

// Spike Array Buffer Configuration (memory leak prevention)
export const SPIKE_BUFFER_CONFIG = {
    MAX_POINTS: 10000, // Maximum points to render (prevents unbounded growth)
    BATCH_SIZE: 40,    // Number of batches for progressive reveal
    ANIMATION_DURATION: 1500 // Animation duration in ms
};

// Legacy global variables (will be migrated to AppState in later phases)
export let currentSignal = null;
export let currentSpikes = null;

// Setters for legacy globals (used by API modules)
export function setCurrentSignal(signal) {
    currentSignal = signal;
}

export function setCurrentSpikes(spikes) {
    currentSpikes = spikes;
}
```

### Phase 1 Testing Checklist

```bash
# Create test file
touch demo/static/js/test-phase1.html
```

```html
<!DOCTYPE html>
<html>
<head>
    <title>Phase 1 Module Test</title>
</head>
<body>
    <h1>Phase 1 Utilities Test</h1>
    <div id="results"></div>

    <script type="module">
        import { detectDeviceCapabilities } from './utils/device.js';
        import { performanceMetrics, measureVisualizationTime } from './utils/performance.js';
        import { formatTime } from './utils/timing.js';
        import { setButtonState, announceToScreenReader } from './utils/accessibility.js';
        import { SPIKE_BUFFER_CONFIG, animationState } from './core/config.js';

        const results = document.getElementById('results');

        // Test 1: Device detection
        detectDeviceCapabilities();
        results.innerHTML += '<p>✓ Device detection loaded</p>';

        // Test 2: Performance metrics
        const testTime = measureVisualizationTime('test', () => {
            let sum = 0;
            for (let i = 0; i < 1000000; i++) sum += i;
            return sum;
        });
        results.innerHTML += `<p>✓ Performance test: ${performanceMetrics.lastUpdate}</p>`;

        // Test 3: Time formatting
        results.innerHTML += `<p>✓ Format time: ${formatTime(1234.56)}</p>`;

        // Test 4: Accessibility
        const btn = document.createElement('button');
        btn.id = 'test-btn';
        setButtonState(btn, true);
        results.innerHTML += `<p>✓ Button disabled: ${btn.disabled}</p>`;

        announceToScreenReader('Test message');
        results.innerHTML += '<p>✓ Screen reader announcement sent</p>';

        // Test 5: Config
        results.innerHTML += `<p>✓ Config loaded: MAX_POINTS=${SPIKE_BUFFER_CONFIG.MAX_POINTS}</p>`;
        results.innerHTML += `<p>✓ Animation state: isInitialized=${animationState.isInitialized}</p>`;

        console.log('✅ Phase 1 modules loaded successfully');
    </script>
</body>
</html>
```

**Test**: Open http://localhost:5000/static/js/test-phase1.html

- [ ] No console errors
- [ ] All 5 checkmarks appear
- [ ] Browser console shows "✅ Phase 1 modules loaded successfully"

### Phase 1 Rollback Procedure

If issues occur:
```bash
# Delete new files
rm -rf demo/static/js/utils demo/static/js/core

# Restore backup (if needed)
# Original script.js remains untouched at this stage
```

### Phase 1 Completion Criteria

- [ ] All 6 new files created
- [ ] Test HTML passes all checks
- [ ] No console errors in browser
- [ ] Git commit: `git commit -m "refactor(frontend): extract Phase 1 utilities and config"`

---

## Phase 2: Core Systems - State, API, Error Handling

**Goal**: Extract state management, API client layer, and error handling
**Time Estimate**: 2 hours
**Risk**: LOW-MEDIUM (central to application, but well-isolated)

### Files to Create

```
demo/static/
└── js/
    ├── core/
    │   └── state.js           # AppState management system
    └── api/
        ├── client.js          # ErrorHandler with retry logic
        ├── health.js          # checkHealth()
        ├── generate.js        # generateSample(), generateSpikes()
        └── predict.js         # runPrediction()
```

### Step-by-Step Instructions

#### 2.1 Extract `js/core/state.js`

**Lines to Extract**: script.js:33-260

```javascript
/**
 * PHASE 4: Centralized State Management System
 *
 * Provides:
 * - Centralized state storage
 * - Type validation
 * - Pub/sub pattern for state changes
 * - State history (for debugging/undo)
 * - Immutability enforcement
 */

/**
 * Application State Management
 */
export const AppState = {
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
```

#### 2.2 Extract `js/api/client.js`

**Lines to Extract**: script.js:263-473

```javascript
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
```

#### 2.3 Extract `js/api/health.js`

**Lines to Extract**: script.js:662-731

```javascript
/**
 * Health Check API
 * Monitors model and device status
 */

import { AppState } from '../core/state.js';
import { ErrorHandler } from './client.js';

/**
 * Check system health and update status UI
 */
export async function checkHealth() {
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
        const modelIndicator = document.getElementById('model-indicator');

        if (data.model.loaded) {
            modelStatus.textContent = 'Loaded';
            modelStatus.className = 'status-badge success';

            // Update model indicator dot to green
            if (modelIndicator) {
                modelIndicator.className = 'label-indicator online';
                modelIndicator.setAttribute('aria-label', 'Model loaded and ready');
            }

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

            // Update model indicator dot to amber/loading
            if (modelIndicator) {
                modelIndicator.className = 'label-indicator loading';
                modelIndicator.setAttribute('aria-label', 'Model not loaded');
            }
        }

        deviceStatus.textContent = data.device.toUpperCase();
        deviceStatus.className = data.device === 'cuda' ? 'status-badge success' : 'status-badge info';

        // Update device memory (VRAM)
        const deviceMemory = document.getElementById('device-memory');
        if (data.device_memory && data.device_memory > 0) {
            deviceMemory.textContent = `${data.device_memory.toFixed(1)} GB VRAM`;
        } else {
            deviceMemory.textContent = 'N/A';
        }

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
```

#### 2.4 Extract `js/api/generate.js`

**Lines to Extract**: script.js:733-847

```javascript
/**
 * Data Generation API
 * Handles ECG signal and spike generation
 */

import { AppState } from '../core/state.js';
import { ErrorHandler } from './client.js';
import { setButtonState } from '../utils/accessibility.js';
import { setCurrentSignal, setCurrentSpikes } from '../core/config.js';
import { measureVisualizationTime } from '../utils/performance.js';

// Forward declarations (will be imported from visualization modules in Phase 3)
let plotECG, plotSpikes, animateCountUp;

/**
 * Set visualization functions (called from main.js after Phase 3)
 */
export function setVisualizationFunctions(functions) {
    ({ plotECG, plotSpikes, animateCountUp } = functions);
}

/**
 * Generate synthetic ECG sample
 */
export async function generateSample() {
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');
    const condition = document.getElementById('condition-select').value;

    try {
        setButtonState(generateBtn, true); // Disable with ARIA sync
        generateBtn.textContent = 'Generating...';

        // CRITICAL: Clean up any ongoing animations before new generation
        // This prevents memory leaks from accumulated animation frames
        const { cleanupAllAnimations } = await import('../animation/cleanup.js');
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

        setCurrentSignal(data.signal);

        // Plot ECG
        if (plotECG) {
            plotECG(data.signal, data.condition);
        }

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

/**
 * Generate spike encoding from signal
 */
export async function generateSpikes(signal) {
    try {
        // Get gain value from AppState or use default
        const gainValue = AppState.getState('ui.gainValue') || 10.0;

        // Measure spike encoding time
        const data = await measureVisualizationTime('spike-encoding', async () => {
            // Use ErrorHandler for resilient fetching with retry logic
            const response = await ErrorHandler.fetchWithRetry('/api/visualize_spikes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    signal: signal,
                    gain: gainValue  // Include custom gain value
                }),
                timeout: 10000 // 10 second timeout for spike encoding
            });

            return await response.json();
        });

        if (data.error) {
            throw new Error(data.error);
        }

        setCurrentSpikes(data);

        // Update spike statistics with animated counters
        const totalSpikes = data.spike_times.length;
        const duration = 10.0; // ECG signal duration is always 10 seconds
        const firingRate = totalSpikes / duration;
        const sparsity = (totalSpikes / (data.num_neurons * data.num_steps)) * 100;

        // Animate all three counters with staggered timing
        if (animateCountUp) {
            animateCountUp(document.getElementById('total-spikes'), 0, totalSpikes, 800);
            animateCountUp(document.getElementById('firing-rate'), 0, firingRate, 800, { suffix: ' Hz', decimals: 1 });
            animateCountUp(document.getElementById('sparsity'), 0, sparsity, 800, { suffix: '%', decimals: 1 });
        }

        if (plotSpikes) {
            plotSpikes(data);
        }

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
```

#### 2.5 Extract `js/api/predict.js`

**Lines to Extract**: script.js:849-898

```javascript
/**
 * Prediction API
 * Handles model inference
 */

import { ErrorHandler } from './client.js';
import { setButtonState } from '../utils/accessibility.js';
import { currentSignal } from '../core/config.js';

// Forward declaration (will be imported from ui modules in Phase 3)
let displayResults;

/**
 * Set UI functions (called from main.js after Phase 3)
 */
export function setUIFunctions(functions) {
    ({ displayResults } = functions);
}

/**
 * Run prediction on current signal
 */
export async function runPrediction() {
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

        if (displayResults) {
            displayResults(data);
        }

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
```

### Phase 2 Testing Checklist

```bash
# Create test file
touch demo/static/js/test-phase2.html
```

```html
<!DOCTYPE html>
<html>
<head>
    <title>Phase 2 Module Test</title>
</head>
<body>
    <h1>Phase 2 Core Systems Test</h1>
    <button id="test-btn">Test Button</button>
    <div id="results"></div>

    <script type="module">
        import { AppState } from './core/state.js';
        import { ErrorHandler } from './api/client.js';
        import { checkHealth } from './api/health.js';

        const results = document.getElementById('results');

        // Test 1: State management
        AppState.setState('model.loaded', true);
        const isLoaded = AppState.getState('model.loaded');
        results.innerHTML += `<p>✓ State: model.loaded = ${isLoaded}</p>`;

        // Test 2: State subscription
        AppState.subscribe('ui.isGenerating', (newVal, oldVal) => {
            results.innerHTML += `<p>✓ Subscription fired: ${oldVal} → ${newVal}</p>`;
        });
        AppState.setState('ui.isGenerating', true);

        // Test 3: Error handler toast
        ErrorHandler.showToast('Test notification', 'info');
        results.innerHTML += '<p>✓ Toast notification shown</p>';

        // Test 4: Fetch with retry (simulate error)
        try {
            await ErrorHandler.fetchWithRetry('/nonexistent-endpoint', { timeout: 1000 });
        } catch (e) {
            results.innerHTML += `<p>✓ Error handled: ${e.message}</p>`;
        }

        console.log('✅ Phase 2 modules loaded successfully');
    </script>
</body>
</html>
```

**Test**: Open http://localhost:5000/static/js/test-phase2.html

- [ ] No console errors
- [ ] State updates work
- [ ] Subscription callback fires
- [ ] Toast appears in bottom-right
- [ ] Error handler catches 404

### Phase 2 Rollback Procedure

```bash
# Delete new files
rm -rf demo/static/js/api
rm demo/static/js/core/state.js

# Test original script.js still works
make demo
```

### Phase 2 Completion Criteria

- [ ] All 5 new files created
- [ ] Test HTML passes all checks
- [ ] State management functional
- [ ] API client handles errors correctly
- [ ] Git commit: `git commit -m "refactor(frontend): extract Phase 2 core systems and API"`

---

## Phase 3: UI & Visualization - Events, Plots, Results

**Goal**: Extract UI event handlers, Plotly visualizations, and results display
**Time Estimate**: 2-2.5 hours
**Risk**: MEDIUM (touches DOM manipulation, plot rendering)

### Files to Create

```
demo/static/
└── js/
    ├── visualization/
    │   ├── plotly-theme.js    # PlotlyTheme configuration
    │   ├── ecg-plot.js        # plotECG()
    │   ├── spike-plot.js      # plotSpikes()
    │   └── plots-init.js      # initializePlots()
    └── ui/
        ├── events.js          # setupEventListeners()
        ├── results.js         # displayResults(), exportResultsAsCSV()
        └── counters.js        # animateCountUp()
```

### Step-by-Step Instructions

#### 3.1 Extract `js/visualization/plotly-theme.js`

**Lines to Extract**: script.js:908-1100

```javascript
/**
 * PHASE 4: Dynamic Plotly Theme System
 *
 * Plotly Theme Configuration - Reads from CSS Variables
 * This creates a centralized theme that stays in sync with CSS
 */

export const PlotlyTheme = {
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
export async function waitForFonts() {
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
```

#### 3.2 Extract `js/visualization/plots-init.js`

**Lines to Extract**: script.js:1122-1158

```javascript
/**
 * Plotly Chart Initialization
 * Sets up empty ECG and spike plots
 */

import { PlotlyTheme, waitForFonts } from './plotly-theme.js';

/**
 * Initialize empty plots with dynamic theme
 */
export function initializePlots() {
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
```

#### 3.3 Extract `js/visualization/ecg-plot.js`

**Lines to Extract**: script.js:1160-1237

```javascript
/**
 * ECG Signal Visualization
 * Plots ECG waveforms with progressive drawing animation
 */

import { PlotlyTheme } from './plotly-theme.js';
import { managedSetTimeout } from '../utils/timing.js';

/**
 * Plot ECG signal with progressive animation
 * @param {Array} signal - ECG signal data
 * @param {string} condition - Condition type (normal/arrhythmia)
 */
export function plotECG(signal, condition) {
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
```

#### 3.4 Extract `js/visualization/spike-plot.js`

**Lines to Extract**: script.js:1239-1358

```javascript
/**
 * Spike Raster Visualization
 * Plots neural spike patterns with progressive reveal animation
 */

import { PlotlyTheme } from './plotly-theme.js';
import { SPIKE_BUFFER_CONFIG } from '../core/config.js';
import { managedSetTimeout } from '../utils/timing.js';

/**
 * Plot spike raster with memory optimizations
 * @param {Object} spikeData - Spike data with spike_times, neuron_ids, num_neurons, num_steps
 */
export function plotSpikes(spikeData) {
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
```

#### 3.5 Extract `js/ui/counters.js`

**Lines to Extract**: script.js:1807-1833

```javascript
/**
 * Number Counter Animations
 * Smooth count-up animations with easing
 */

/**
 * Animate count-up from start to end value
 * @param {HTMLElement} element - Element to animate
 * @param {number} start - Starting value
 * @param {number} end - Ending value
 * @param {number} duration - Animation duration in ms
 * @param {Object} options - { suffix, decimals }
 */
export function animateCountUp(element, start, end, duration = 1000, options = {}) {
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
```

#### 3.6 Extract `js/ui/events.js`

**Lines to Extract**: script.js:579-656

```javascript
/**
 * Event Listeners Setup
 * Handles all DOM event bindings
 */

import { generateSample } from '../api/generate.js';
import { runPrediction } from '../api/predict.js';
import { exportResultsAsCSV } from './results.js';
import { AppState } from '../core/state.js';
import { ErrorHandler } from '../api/client.js';

/**
 * Setup all event listeners for the application
 */
export function setupEventListeners() {
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

    // Gain slider event listener
    const gainSlider = document.getElementById('gain-slider');
    const gainValue = document.getElementById('gain-value');
    const gainPreview = document.getElementById('gain-preview');

    if (gainSlider) {
        gainSlider.addEventListener('input', function(e) {
            const gain = parseFloat(e.target.value);
            gainValue.textContent = gain.toFixed(1);

            // Update preview text based on gain value
            if (gain < 5) {
                gainPreview.textContent = 'Low spike rate (sparse)';
            } else if (gain < 10) {
                gainPreview.textContent = 'Below-normal spike rate';
            } else if (gain === 10) {
                gainPreview.textContent = 'Normal spike rate';
            } else if (gain < 15) {
                gainPreview.textContent = 'Above-normal spike rate';
            } else {
                gainPreview.textContent = 'High spike rate (dense)';
            }

            // Store in AppState for later use
            AppState.setState('ui.gainValue', gain);
        });
    }

    // Export button event listeners
    const exportEcgBtn = document.getElementById('export-ecg-btn');
    const exportSpikesBtn = document.getElementById('export-spikes-btn');
    const exportCsvBtn = document.getElementById('export-csv-btn');

    if (exportEcgBtn) {
        exportEcgBtn.addEventListener('click', () => {
            Plotly.downloadImage('ecg-plot', {
                format: 'png',
                width: 1200,
                height: 600,
                filename: 'cortexcore_ecg.png'
            });
            ErrorHandler.showToast('ECG plot downloaded', 'success');
        });
    }

    if (exportSpikesBtn) {
        exportSpikesBtn.addEventListener('click', () => {
            Plotly.downloadImage('spike-plot', {
                format: 'png',
                width: 1200,
                height: 600,
                filename: 'cortexcore_spikes.png'
            });
            ErrorHandler.showToast('Spike raster downloaded', 'success');
        });
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', exportResultsAsCSV);
    }
}
```

#### 3.7 Extract `js/ui/results.js`

**Lines to Extract**: script.js:1364-1496

```javascript
/**
 * Results Display and Export
 * Handles prediction results UI and CSV export
 */

import { ErrorHandler } from '../api/client.js';
import { announceToScreenReader } from '../utils/accessibility.js';
import { currentSignal, currentSpikes } from '../core/config.js';

/**
 * Display prediction results in UI
 * @param {Object} results - Prediction results
 */
export function displayResults(results) {
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

    // Update inference timestamp
    const inferenceTimestamp = document.getElementById('inference-timestamp');
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
    });
    inferenceTimestamp.textContent = timeStr;

    // Update spikes processed
    const spikesProcessed = document.getElementById('spikes-processed');
    if (results.spike_count !== undefined) {
        spikesProcessed.textContent = `${results.spike_count.toLocaleString()} spikes`;
    } else {
        spikesProcessed.textContent = '-';
    }

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
 * Export results as CSV file
 */
export function exportResultsAsCSV() {
    // Get current state
    const signal = currentSignal;
    const spikes = currentSpikes;

    if (!signal) {
        ErrorHandler.showToast('No signal data to export. Generate a signal first.', 'warning');
        return;
    }

    // Get results data from the DOM
    const classificationText = document.getElementById('prediction-class').textContent;
    const confidenceText = document.getElementById('prediction-confidence').textContent;
    const inferenceTimeText = document.getElementById('inference-time').textContent;
    const conditionSelect = document.getElementById('condition-select');
    const condition = conditionSelect ? conditionSelect.value : 'Unknown';

    // Get spike stats if available
    const totalSpikesText = document.getElementById('total-spikes').textContent || '0';
    const firingRateText = document.getElementById('firing-rate').textContent || '0 Hz';
    const sparsityText = document.getElementById('sparsity').textContent || '0%';

    // Prepare CSV content
    const lines = [
        ['Metric', 'Value'],
        ['Condition', condition.charAt(0).toUpperCase() + condition.slice(1)],
        ['Classification', classificationText || 'N/A'],
        ['Confidence', confidenceText || 'N/A'],
        ['Inference Time', inferenceTimeText || 'N/A'],
        ['Timestamp', new Date().toLocaleString()],
        [],
        ['Signal Statistics'],
        ['Duration', '10 seconds'],
        ['Sampling Rate', '250 Hz'],
        ['Samples', signal.length],
        [],
        ['Spike Statistics'],
        ['Total Spikes', totalSpikesText],
        ['Firing Rate', firingRateText],
        ['Sparsity', sparsityText],
    ];

    const csvContent = lines
        .map(row => row.map(cell => `"${cell}"`).join(','))
        .join('\n');

    // Create and download file
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cortexcore_results_${new Date().getTime()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    ErrorHandler.showToast('Results exported as CSV', 'success');
}
```

### Phase 3 Testing Checklist

**Manual Testing** (demo must be running):

```bash
# Start demo
make demo
```

Navigate to http://localhost:5000:

- [ ] Generate ECG sample (test both normal and arrhythmia)
- [ ] Verify ECG plot appears with progressive drawing animation
- [ ] Adjust gain slider and verify preview text updates
- [ ] Verify spike plot appears with progressive reveal
- [ ] Check spike statistics counters animate
- [ ] Run prediction and verify results display
- [ ] Verify probability bars animate correctly
- [ ] Test export buttons (ECG, spikes, CSV)
- [ ] Check toast notifications appear
- [ ] Verify no console errors

### Phase 3 Rollback Procedure

```bash
# Delete new files
rm -rf demo/static/js/visualization demo/static/js/ui

# Test original script.js still works
make demo
```

### Phase 3 Completion Criteria

- [ ] All 7 new files created
- [ ] Demo UI fully functional
- [ ] Plots render correctly
- [ ] Animations work smoothly
- [ ] Export functionality works
- [ ] Git commit: `git commit -m "refactor(frontend): extract Phase 3 UI and visualization"`

---

## Phase 4: Animation Systems & Final Integration

**Goal**: Extract animation modules, create main entry point, update HTML
**Time Estimate**: 2 hours
**Risk**: MEDIUM-HIGH (final integration, critical path)

### Files to Create

```
demo/static/
├── js/
│   ├── animation/
│   │   ├── system.js          # initializeAnimationSystems()
│   │   ├── particles.js       # Floating particles
│   │   ├── cleanup.js         # Memory leak prevention
│   │   ├── scroll.js          # Intersection Observer
│   │   └── performance.js     # Performance monitoring (debug mode)
│   └── main.js                # Entry point: imports + initializes
└── css/
    └── main.css               # @import all CSS modules
```

### Step-by-Step Instructions

#### 4.1 Extract `js/animation/cleanup.js`

**Lines to Extract**: script.js:521-573

```javascript
/**
 * Animation Cleanup Functions (Memory Leak Prevention)
 * Manages lifecycle of timeouts and animation frames
 */

import { animationState } from '../core/config.js';

/**
 * Clear all active timeouts and animation frames
 * Critical for preventing memory leaks
 */
export function cleanupAllAnimations() {
    // Cancel all setTimeout calls
    animationState.activeTimeouts.forEach(id => clearTimeout(id));
    animationState.activeTimeouts = [];

    // Cancel all requestAnimationFrame calls
    animationState.activeAnimationFrames.forEach(id => cancelAnimationFrame(id));
    animationState.activeAnimationFrames = [];

    console.log('🧹 Cleaned up all active animations');
}
```

#### 4.2 Extract `js/animation/particles.js`

**Lines to Extract**: script.js:1702-1785

```javascript
/**
 * Floating Particles System
 * Creates neural network-inspired particle effects
 */

import { animationState } from '../core/config.js';

/**
 * Create floating particles with proper lifecycle management
 * MEMORY LEAK FIX: Store particle references for cleanup
 * @param {number} count - Number of particles to create
 */
export function createFloatingParticles(count) {
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
export function clearFloatingParticles() {
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
export function pauseParticleAnimations() {
    animationState.particles.forEach(particle => {
        particle.style.animationPlayState = 'paused';
    });
}

/**
 * Resume particle animations (called when tab is visible)
 */
export function resumeParticleAnimations() {
    animationState.particles.forEach(particle => {
        particle.style.animationPlayState = 'running';
    });
}
```

#### 4.3 Extract `js/animation/scroll.js`

**Lines to Extract**: script.js:1631-1660

```javascript
/**
 * Scroll-Triggered Animations (Intersection Observer)
 * Animates elements when they enter viewport
 */

/**
 * Initialize Intersection Observer for scroll animations
 */
export function initializeScrollAnimations() {
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
```

#### 4.4 Extract `js/animation/performance.js`

**Lines to Extract**: script.js:1897-1949

```javascript
/**
 * Performance Monitoring (Debug Mode)
 * FPS counter and memory usage display
 */

import { performanceMetrics } from '../utils/performance.js';

/**
 * Setup performance monitoring overlay
 * Only activates in debug mode (?debug in URL)
 */
export function setupPerformanceMonitoring() {
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
```

#### 4.5 Extract `js/animation/system.js`

**Lines to Extract**: script.js:1562-1594, 1667-1699, 1792-1800, 2009-2035

```javascript
/**
 * PHASE 3: ANIMATION SYSTEMS
 * Centralized animation initialization and management
 */

import { detectDeviceCapabilities } from '../utils/device.js';
import { initializeScrollAnimations } from './scroll.js';
import { createFloatingParticles, pauseParticleAnimations, resumeParticleAnimations } from './particles.js';
import { setupPerformanceMonitoring } from './performance.js';

/**
 * Initialize all animation systems
 */
export function initializeAnimationSystems() {
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

/**
 * Setup button ripple effects
 */
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

/**
 * Create grid scanner effect
 */
function createGridScanner() {
    // Create scanner if it doesn't exist
    if (!document.querySelector('.grid-scanner')) {
        const scanner = document.createElement('div');
        scanner.classList.add('grid-scanner');
        document.body.appendChild(scanner);
        console.log('📡 Grid scanner effect enabled');
    }
}

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
```

#### 4.6 Create `js/main.js` (ENTRY POINT)

```javascript
/**
 * CortexCore Demo - Main Entry Point
 *
 * Coordinates all modules and initializes the application
 *
 * Module Loading Order:
 * 1. Core (state, config)
 * 2. Utils (device, performance, timing, accessibility)
 * 3. API (client, health, generate, predict)
 * 4. UI (events, results, counters)
 * 5. Visualization (plotly-theme, plots)
 * 6. Animation (system, particles, cleanup)
 */

// ============================================
// Core Systems
// ============================================
import { AppState } from './core/state.js';
import { animationState, SPIKE_BUFFER_CONFIG } from './core/config.js';
import { ErrorHandler } from './api/client.js';

// ============================================
// API Layer
// ============================================
import { checkHealth } from './api/health.js';
import { generateSample, setVisualizationFunctions as setGenVisualizationFunctions } from './api/generate.js';
import { runPrediction, setUIFunctions as setPredictUIFunctions } from './api/predict.js';

// ============================================
// UI Layer
// ============================================
import { setupEventListeners } from './ui/events.js';
import { displayResults } from './ui/results.js';
import { animateCountUp } from './ui/counters.js';

// ============================================
// Visualization Layer
// ============================================
import { initializePlots } from './visualization/plots-init.js';
import { plotECG } from './visualization/ecg-plot.js';
import { plotSpikes } from './visualization/spike-plot.js';

// ============================================
// Animation Layer
// ============================================
import { initializeAnimationSystems } from './animation/system.js';
import { cleanupAllAnimations } from './animation/cleanup.js';

// ============================================
// Wire up cross-module dependencies
// ============================================

// API modules need visualization functions
setGenVisualizationFunctions({ plotECG, plotSpikes, animateCountUp });

// Prediction module needs results display
setPredictUIFunctions({ displayResults });

// ============================================
// Expose globals for HTML onclick handlers
// (Temporary - migrate to data-* attributes later)
// ============================================
window.AppState = AppState;
window.ErrorHandler = ErrorHandler;
window.generateSample = generateSample;
window.runPrediction = runPrediction;

// ============================================
// Application Initialization
// ============================================

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

    console.log('✅ All systems operational');
});

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
```

#### 4.7 Create CSS Module Structure

**Create directory structure:**
```bash
cd demo/static
mkdir -p css/base css/components css/animations css/utilities
```

**Split style.css** (I'll provide extraction script below):

```bash
# Create extraction script
cat > split-css.sh << 'EOF'
#!/bin/bash

CSS_FILE="style.css"
BASE_DIR="css"

# Extract base styles
sed -n '1,113p' $CSS_FILE > $BASE_DIR/base/variables.css      # CSS variables
sed -n '116,140p' $CSS_FILE > $BASE_DIR/base/reset.css        # Reset & body
sed -n '177,212p' $CSS_FILE > $BASE_DIR/base/typography.css   # Typography
sed -n '215,227p' $CSS_FILE > $BASE_DIR/base/layout.css       # Container

# Extract components (27 sections)
sed -n '230,282p' $CSS_FILE > $BASE_DIR/components/header.css
sed -n '284,361p' $CSS_FILE > $BASE_DIR/components/cards.css
sed -n '363,461p' $CSS_FILE > $BASE_DIR/components/status.css
sed -n '464,513p' $CSS_FILE > $BASE_DIR/components/controls.css
sed -n '515,591p' $CSS_FILE > $BASE_DIR/components/buttons.css
sed -n '593,627p' $CSS_FILE > $BASE_DIR/components/plots.css
sed -n '629,698p' $CSS_FILE > $BASE_DIR/components/results.css
sed -n '700,792p' $CSS_FILE > $BASE_DIR/components/probability.css

# Extract animations
sed -n '875,930p' $CSS_FILE > $BASE_DIR/animations/loading.css
sed -n '932,960p' $CSS_FILE > $BASE_DIR/animations/keyframes.css

# Extract utilities
sed -n '963,1016p' $CSS_FILE > $BASE_DIR/utilities/accessibility.css
sed -n '1018,1099p' $CSS_FILE > $BASE_DIR/utilities/responsive.css
sed -n '1102,1142p' $CSS_FILE > $BASE_DIR/utilities/helpers.css

echo "✅ CSS split complete"
EOF

chmod +x split-css.sh
./split-css.sh
```

**Create `css/main.css`:**

```css
/**
 * CortexCore - Main CSS Entry Point
 * Imports all modular stylesheets
 */

/* ============================================
   BASE STYLES
   ============================================ */
@import './base/variables.css';
@import './base/reset.css';
@import './base/typography.css';
@import './base/layout.css';

/* ============================================
   COMPONENTS
   ============================================ */
@import './components/header.css';
@import './components/cards.css';
@import './components/status.css';
@import './components/controls.css';
@import './components/buttons.css';
@import './components/plots.css';
@import './components/results.css';
@import './components/probability.css';

/* ============================================
   ANIMATIONS
   ============================================ */
@import './animations/keyframes.css';
@import './animations/loading.css';

/* ============================================
   UTILITIES
   ============================================ */
@import './utilities/accessibility.css';
@import './utilities/responsive.css';
@import './utilities/helpers.css';
```

#### 4.8 Update HTML Templates

**Edit `demo/templates/index.html`:**

Find the `<head>` section and replace script/style tags:

```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CortexCore - SNN ECG Demo</title>

    <!-- OLD (keep commented for rollback) -->
    <!-- <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="{{ url_for('static', filename='script.js') }}"></script> -->

    <!-- NEW MODULAR IMPORTS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">

    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>

    <!-- Main Application (ES6 Module) -->
    <script type="module" src="{{ url_for('static', filename='js/main.js') }}"></script>

    <!-- Cache busting (optional, for development) -->
    <!-- <script type="module" src="{{ url_for('static', filename='js/main.js') }}?v=<%= Date.now() %>"></script> -->
</head>
```

### Phase 4 Testing Checklist

**Full Application Testing:**

```bash
# Start demo
make demo
```

Navigate to http://localhost:5000:

#### Functionality Tests
- [ ] **Health Check**: Model status loads correctly
- [ ] **ECG Generation**:
  - [ ] Normal condition generates
  - [ ] Arrhythmia condition generates
  - [ ] ECG plot renders with animation
- [ ] **Spike Encoding**:
  - [ ] Spike raster appears
  - [ ] Statistics counters animate
  - [ ] Gain slider updates preview text
- [ ] **Prediction**:
  - [ ] Prediction runs successfully
  - [ ] Results card displays
  - [ ] Probability bars animate
- [ ] **Export**:
  - [ ] ECG export works
  - [ ] Spike export works
  - [ ] CSV export works

#### Visual/Animation Tests
- [ ] **Particles**: Floating particles visible (desktop only)
- [ ] **Ripples**: Button clicks show ripple effect
- [ ] **Grid Scanner**: Background scanner animates
- [ ] **Scroll Animations**: Cards animate on scroll
- [ ] **Toasts**: Notifications appear and disappear
- [ ] **Page Load**: Staggered card reveals on initial load

#### Technical Tests
- [ ] **Console**: No JavaScript errors
- [ ] **Network**: All modules load (check Network tab)
- [ ] **Performance**: FPS stable (add `?debug` to URL)
- [ ] **Memory**: No memory leaks (Profile tab, heap snapshots)
- [ ] **Mobile**: Responsive on small screens
- [ ] **Accessibility**: Keyboard navigation works

#### Browser Compatibility
- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari (if available)

### Phase 4 Rollback Procedure

If critical issues occur:

```bash
# Edit templates/index.html - uncomment old script tags
# Comment out new modular imports

# Restart Flask
make demo
```

Full rollback:
```bash
# Restore original files
cp -r demo/static.backup/* demo/static/

# Or use git
git checkout -- demo/static/ demo/templates/index.html
```

### Phase 4 Completion Criteria

- [ ] All 6 animation files created
- [ ] main.js entry point created
- [ ] CSS modules created (25+ files)
- [ ] HTML updated to use modules
- [ ] Full application functional
- [ ] All tests pass
- [ ] No console errors
- [ ] Git commit: `git commit -m "refactor(frontend): complete Phase 4 animation systems and modularization"`

---

## Post-Refactoring Tasks

### Cleanup

```bash
# Remove backup files (after verification)
rm -rf demo/static.backup

# Remove test files
rm demo/static/js/test-phase1.html
rm demo/static/js/test-phase2.html

# Remove original monolithic files (keep for 1-2 weeks for safety)
# mv demo/static/script.js demo/static/script.js.old
# mv demo/static/style.css demo/static/style.css.old
```

### Documentation Updates

Create `demo/static/README.md`:

```markdown
# CortexCore Demo - Frontend Structure

## Directory Organization

- `js/` - JavaScript modules (ES6)
  - `core/` - State management, configuration
  - `utils/` - Utility functions (device, performance, accessibility)
  - `api/` - Backend API clients (health, generate, predict)
  - `ui/` - User interface (events, results, counters)
  - `visualization/` - Plotly charts (ECG, spikes)
  - `animation/` - Animation systems (particles, scroll, performance)
  - `main.js` - Entry point

- `css/` - Modular stylesheets
  - `base/` - Variables, reset, typography, layout
  - `components/` - UI components (cards, buttons, plots, etc.)
  - `animations/` - Keyframes and loading states
  - `utilities/` - Accessibility, responsive design, helpers
  - `main.css` - Import orchestrator

## Development Workflow

### Adding New Features

1. **New API Endpoint**: Add to `js/api/`
2. **New UI Component**:
   - JS: Add to `js/ui/`
   - CSS: Add to `css/components/`
3. **New Animation**: Add to `js/animation/`

### Modifying Existing Code

1. Navigate to specific module (e.g., `js/api/generate.js`)
2. Make changes
3. Test in browser (hard refresh: Ctrl+Shift+R)
4. Commit changes

### Testing

- **Unit Test**: Import module in test file
- **Integration Test**: Run full demo, test feature
- **Performance**: Add `?debug` to URL for FPS counter

## Browser Support

- Chrome/Edge 61+ (ES6 modules)
- Firefox 60+
- Safari 11+
- No IE11 support (ES6 modules required)

## Troubleshooting

### Module Not Loading
1. Check browser console for 404 errors
2. Verify import path is correct (case-sensitive)
3. Check file exists in `js/` directory

### Visual Regression
1. Check CSS @import order in `css/main.css`
2. Verify specificity hasn't changed
3. Clear browser cache

### Performance Issues
1. Enable debug mode: `?debug` in URL
2. Check FPS counter (top-right)
3. Profile memory in DevTools

## Migration Notes

Refactored from monolithic `script.js` (2076 lines) and `style.css` (2000+ lines) on 2025-11-12.

Original files preserved as:
- `script.js.old`
- `style.css.old`
```

### Git Workflow

```bash
# Final commit
git add demo/static demo/templates/index.html
git commit -m "refactor(frontend): complete modularization

- Split script.js (2076 lines) into 20+ modules
- Split style.css (2000+ lines) into 25+ modules
- Created ES6 module architecture
- Improved developer experience and maintainability
- Zero breaking changes to functionality

Phases completed:
1. Foundation (utils, config)
2. Core systems (state, API, error handling)
3. UI & visualization (events, plots, results)
4. Animation systems & integration

Benefits:
- Parallel development (no merge conflicts)
- Better browser caching (granular invalidation)
- Easier testing (import single modules)
- Cleaner git diffs (specific file changes)
- Future-ready for code splitting and tree shaking
"

# Create PR (if using GitHub)
git push origin refactor/frontend-modularization

# Or merge to main if confident
git checkout main
git merge refactor/frontend-modularization
git push origin main
```

---

## Success Metrics

### Before Refactoring
- **Files**: 2 (script.js, style.css)
- **Lines**: 4,076 total
- **Merge Conflicts**: Frequent (multi-person edits to same files)
- **Navigation**: Ctrl+F + line numbers
- **Testing**: Import entire 2MB script
- **Cache Invalidation**: Entire 2MB on any change

### After Refactoring
- **Files**: 45+ (20 JS modules, 25 CSS modules)
- **Lines**: 4,076 total (same logic, better organized)
- **Merge Conflicts**: Rare (isolated file changes)
- **Navigation**: Jump to specific module
- **Testing**: Import single module (~50-200 lines)
- **Cache Invalidation**: Only changed modules (~10-100KB)

### Developer Experience Improvements
- ✅ **Parallel Work**: 3+ developers can work simultaneously
- ✅ **Faster Debugging**: Jump to specific module in <2 seconds
- ✅ **Better Git Diffs**: "Modified api/generate.js" vs "Modified script.js (lines 733-847)"
- ✅ **Easier Onboarding**: New developers understand structure quickly
- ✅ **Future-Proof**: Ready for bundlers (webpack, rollup), tree shaking, code splitting

---

## Appendix A: Quick Reference

### Module Import Cheatsheet

```javascript
// Core
import { AppState } from './core/state.js';
import { ErrorHandler } from './api/client.js';

// API
import { checkHealth } from './api/health.js';
import { generateSample } from './api/generate.js';
import { runPrediction } from './api/predict.js';

// UI
import { setupEventListeners } from './ui/events.js';
import { displayResults } from './ui/results.js';
import { animateCountUp } from './ui/counters.js';

// Visualization
import { PlotlyTheme } from './visualization/plotly-theme.js';
import { plotECG } from './visualization/ecg-plot.js';
import { plotSpikes } from './visualization/spike-plot.js';

// Animation
import { initializeAnimationSystems } from './animation/system.js';
import { cleanupAllAnimations } from './animation/cleanup.js';

// Utils
import { setButtonState, announceToScreenReader } from './utils/accessibility.js';
import { formatTime, managedSetTimeout } from './utils/timing.js';
import { detectDeviceCapabilities } from './utils/device.js';
```

### CSS @import Cheatsheet

```css
/* Base */
@import './base/variables.css';
@import './base/reset.css';
@import './base/typography.css';
@import './base/layout.css';

/* Components */
@import './components/header.css';
@import './components/cards.css';
@import './components/buttons.css';
/* ... etc ... */

/* Animations */
@import './animations/keyframes.css';
@import './animations/loading.css';

/* Utilities */
@import './utilities/accessibility.css';
@import './utilities/responsive.css';
@import './utilities/helpers.css';
```

### File Location Quick Lookup

| Need to... | Edit this file... |
|------------|-------------------|
| Add new API endpoint | `js/api/*.js` |
| Change state management | `js/core/state.js` |
| Modify ECG plot | `js/visualization/ecg-plot.js` |
| Update spike visualization | `js/visualization/spike-plot.js` |
| Add event listener | `js/ui/events.js` |
| Change button styles | `css/components/buttons.css` |
| Modify color theme | `css/base/variables.css` |
| Add animation | `js/animation/*.js` |
| Update toast notifications | `js/api/client.js` (ErrorHandler) |
| Fix accessibility issue | `js/utils/accessibility.js` |

---

## Appendix B: Troubleshooting Guide

### Issue: Module not found (404)
**Symptom**: Console error "Failed to load module script: The server responded with a non-JavaScript MIME type"

**Solution**:
1. Verify file exists: `ls demo/static/js/path/to/module.js`
2. Check import path is correct (case-sensitive on Linux)
3. Ensure Flask is serving `/static/` correctly
4. Check browser Network tab for actual request URL

### Issue: Circular dependency
**Symptom**: "Cannot access 'X' before initialization"

**Solution**:
1. Identify circular import: Module A imports B, B imports A
2. Refactor to break cycle (e.g., move shared code to new module)
3. Use dynamic import for one direction: `const { X } = await import('./module.js');`

### Issue: Global variable undefined
**Symptom**: "ReferenceError: currentSignal is not defined"

**Solution**:
1. Check module exports/imports: `export const currentSignal = null;`
2. Use getter functions instead of direct access:
   ```javascript
   // config.js
   export function getCurrentSignal() { return currentSignal; }

   // usage
   import { getCurrentSignal } from './core/config.js';
   const signal = getCurrentSignal();
   ```

### Issue: CSS styles not applying
**Symptom**: Visual regression, missing styles

**Solution**:
1. Check `css/main.css` import order (specificity matters)
2. Verify CSS file exists and is imported
3. Hard refresh browser: Ctrl+Shift+R
4. Check browser DevTools → Network tab → CSS files loaded

### Issue: Animation not working
**Symptom**: Particles missing, counters not animating

**Solution**:
1. Check `js/animation/system.js` is called
2. Verify `initializeAnimationSystems()` runs on DOMContentLoaded
3. Check console for animation-related errors
4. Test on desktop (particles disabled on mobile)

---

## Appendix C: Future Enhancements

### Potential Phase 5 (Future)

1. **Bundler Integration**
   - Add webpack/rollup for production builds
   - Enable tree shaking (remove unused code)
   - Code splitting (load modules on-demand)
   - Minification and compression

2. **TypeScript Migration**
   - Add type safety to all modules
   - Improve IDE autocomplete
   - Catch errors at compile-time

3. **Testing Infrastructure**
   - Unit tests (Jest/Vitest)
   - Integration tests (Playwright)
   - Visual regression tests

4. **Advanced Optimizations**
   - Service worker for offline support
   - WebAssembly for performance-critical code
   - Web Workers for background processing

5. **Developer Tools**
   - Hot module replacement (HMR)
   - Source maps for debugging
   - Linting and formatting automation
   - Pre-commit hooks

---

**End of Refactoring Plan**
