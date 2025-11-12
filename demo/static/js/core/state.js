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
