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
