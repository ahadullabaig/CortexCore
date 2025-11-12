/**
 * Spike Raster Visualization
 * Plots neural spike patterns with progressive reveal animation
 */

import { PlotlyTheme } from './plotly-theme.js';
import { SPIKE_BUFFER_CONFIG } from '../core/config.js';
import { managedSetTimeout } from '../utils/timing.js';

/**
 * Plot spike raster with memory optimizations
 * @param {Object} spikeData - Spike data with spike_times, neuron_ids, num_neurons, num_steps
 */
export function plotSpikes(spikeData) {
    // MEMORY LEAK FIX: Implement data subsampling if spike count exceeds buffer limit
    const totalSpikes = spikeData.spike_times.length;
    let displayData = { spike_times: spikeData.spike_times, neuron_ids: spikeData.neuron_ids };

    if (totalSpikes > SPIKE_BUFFER_CONFIG.MAX_POINTS) {
        console.warn(`⚠️ Spike count (${totalSpikes}) exceeds buffer limit (${SPIKE_BUFFER_CONFIG.MAX_POINTS}). Subsampling for performance.`);

        // Subsample uniformly to stay within buffer limit
        const step = Math.ceil(totalSpikes / SPIKE_BUFFER_CONFIG.MAX_POINTS);
        displayData = {
            spike_times: spikeData.spike_times.filter((_, idx) => idx % step === 0),
            neuron_ids: spikeData.neuron_ids.filter((_, idx) => idx % step === 0)
        };
    }

    // Use PlotlyTheme for consistent spike color
    const spikeColor = PlotlyTheme.getSpikeColor();

    const trace = {
        x: displayData.spike_times,
        y: displayData.neuron_ids,
        mode: 'markers',
        type: 'scatter',
        name: 'Neural Spikes',
        marker: {
            color: spikeColor,
            size: 5,
            symbol: 'line-ns-open',
            line: {
                width: 2,
                color: spikeColor
            },
            opacity: 0.9
        },
        hovertemplate: '<b>Time</b>: %{x}<br><b>Neuron</b>: %{y}<extra></extra>'
    };

    // Get base layout with dynamic theme
    const baseLayout = PlotlyTheme.getBaseLayout(
        `Spike Raster Plot - ${totalSpikes.toLocaleString()} spikes${totalSpikes > SPIKE_BUFFER_CONFIG.MAX_POINTS ? ' (subsampled)' : ''}`
    );

    const layout = {
        ...baseLayout,
        xaxis: PlotlyTheme.getAxisConfig('Time Step', {
            range: [0, spikeData.num_steps]
        }),
        yaxis: PlotlyTheme.getAxisConfig('Neuron Index', {
            range: [-1, spikeData.num_neurons]
        }),
        showlegend: false // Single trace, no need for legend
    };

    // Use accessible config with descriptive button labels
    Plotly.newPlot('spike-plot', [trace], layout, PlotlyTheme.getAccessibleConfig());

    // Add progressive reveal animation for temporal dynamics visualization
    // Group spikes by time step
    const spikesByTime = {};
    for (let i = 0; i < displayData.spike_times.length; i++) {
        const timeStep = displayData.spike_times[i];
        if (!spikesByTime[timeStep]) {
            spikesByTime[timeStep] = { times: [], neurons: [] };
        }
        spikesByTime[timeStep].times.push(timeStep);
        spikesByTime[timeStep].neurons.push(displayData.neuron_ids[i]);
    }

    const timeSteps = Object.keys(spikesByTime).map(Number).sort((a, b) => a - b);

    // Progressive reveal: gradually show spikes based on their temporal order
    // MEMORY LEAK FIX: Use Plotly.extendTraces instead of accumulating arrays
    const numBatches = Math.min(timeSteps.length, SPIKE_BUFFER_CONFIG.BATCH_SIZE);
    const batchSize = Math.ceil(timeSteps.length / numBatches);
    const delayPerBatch = SPIKE_BUFFER_CONFIG.ANIMATION_DURATION / numBatches;

    let currentBatch = 0;

    // Initialize with empty data for progressive reveal
    Plotly.restyle('spike-plot', {
        x: [[]],
        y: [[]]
    }, [0]);

    function revealNextBatch() {
        if (currentBatch >= numBatches) return;

        // Calculate batch data without accumulating in closure
        const startIdx = currentBatch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, timeSteps.length);

        const batchX = [];
        const batchY = [];

        for (let i = startIdx; i < endIdx; i++) {
            const timeStep = timeSteps[i];
            const spikes = spikesByTime[timeStep];
            batchX.push(...spikes.times);
            batchY.push(...spikes.neurons);
        }

        // Use extendTraces instead of restyle for better performance
        // This appends data instead of replacing entire trace
        Plotly.extendTraces('spike-plot', {
            x: [batchX],
            y: [batchY]
        }, [0]);

        currentBatch++;

        if (currentBatch < numBatches) {
            // Use managed timeout for cleanup tracking
            managedSetTimeout(revealNextBatch, delayPerBatch);
        }
    }

    // Start progressive reveal after brief delay
    managedSetTimeout(revealNextBatch, 200);
}
