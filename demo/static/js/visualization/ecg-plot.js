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
