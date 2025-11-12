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
