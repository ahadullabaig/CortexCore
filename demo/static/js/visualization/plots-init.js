/**
 * Plotly Chart Initialization
 * Sets up empty ECG and spike plots
 */

import { PlotlyTheme, waitForFonts } from './plotly-theme.js';

/**
 * Initialize empty plots with dynamic theme
 */
export function initializePlots() {
    // Wait for fonts before rendering
    waitForFonts().then(() => {
        // Get base theme configuration
        const baseLayout = PlotlyTheme.getBaseLayout('No signal generated yet');

        // Initialize empty ECG plot with dynamic theme
        const ecgLayout = {
            ...baseLayout,
            title: {
                ...baseLayout.title,
                text: 'No signal generated yet'
            },
            xaxis: PlotlyTheme.getAxisConfig('Sample'),
            yaxis: PlotlyTheme.getAxisConfig('Amplitude')
        };

        // Use accessible config for better WCAG compliance
        Plotly.newPlot('ecg-plot', [], ecgLayout, PlotlyTheme.getAccessibleConfig());

        // Initialize empty spike plot with dynamic theme
        const spikeLayout = {
            ...baseLayout,
            title: {
                ...baseLayout.title,
                text: 'No spikes generated yet'
            },
            xaxis: PlotlyTheme.getAxisConfig('Time Step'),
            yaxis: PlotlyTheme.getAxisConfig('Neuron Index')
        };

        // Use accessible config for better WCAG compliance
        Plotly.newPlot('spike-plot', [], spikeLayout, PlotlyTheme.getAccessibleConfig());

        console.log('📊 Plotly charts initialized with dynamic CSS theme');
    });
}
