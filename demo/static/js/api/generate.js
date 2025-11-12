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
let plotECG, plotSpikes, animateCountUp, cleanupAllAnimations;

/**
 * Set visualization functions (called from main.js after Phase 3)
 */
export function setVisualizationFunctions(functions) {
    ({ plotECG, plotSpikes, animateCountUp, cleanupAllAnimations } = functions);
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
        if (cleanupAllAnimations) {
            cleanupAllAnimations();
        }

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
