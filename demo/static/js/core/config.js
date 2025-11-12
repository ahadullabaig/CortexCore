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
