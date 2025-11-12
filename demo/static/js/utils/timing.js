/**
 * Timing Utilities
 * Time formatting and managed timeout/animation frame tracking
 */

// Import animationState at module level (circular dependency is safe in ES6 modules)
import { animationState } from '../core/config.js';

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
