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
