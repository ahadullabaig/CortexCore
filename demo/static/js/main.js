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
setGenVisualizationFunctions({ plotECG, plotSpikes, animateCountUp, cleanupAllAnimations });

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
