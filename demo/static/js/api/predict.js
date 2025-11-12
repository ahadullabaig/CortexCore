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
