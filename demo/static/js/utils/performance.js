/**
 * Performance Monitoring Utilities
 * Tracks render times, memory usage, and frame rates
 */

// Global performance metrics storage
export const performanceMetrics = {
    ecgRenderTime: 0,
    spikeRenderTime: 0,
    spikeEncodingTime: 0,
    lastUpdate: Date.now()
};

/**
 * Measure and display visualization performance
 * @param {string} name - Metric name (e.g., 'ecg-render', 'spike-encoding')
 * @param {function} callback - Function to measure (sync or async)
 * @returns {*} Result of callback
 */
export function measureVisualizationTime(name, callback) {
    const startMark = `${name}-start`;
    const endMark = `${name}-end`;
    const measureName = `${name}-duration`;

    performance.mark(startMark);

    // Execute the callback (can be sync or async)
    const result = callback();

    // Handle both sync and async callbacks
    if (result && typeof result.then === 'function') {
        return result.then((value) => {
            performance.mark(endMark);
            const measure = performance.measure(measureName, startMark, endMark);
            updatePerformanceMetrics(name, measure.duration);
            return value;
        });
    } else {
        performance.mark(endMark);
        const measure = performance.measure(measureName, startMark, endMark);
        updatePerformanceMetrics(name, measure.duration);
        return result;
    }
}

/**
 * Update stored metrics (display updated by FPS loop in debug mode)
 * @private
 */
function updatePerformanceMetrics(name, duration) {
    if (name.includes('ecg')) {
        performanceMetrics.ecgRenderTime = duration;
    } else if (name.includes('spike-encoding')) {
        performanceMetrics.spikeEncodingTime = duration;
    } else if (name.includes('spike')) {
        performanceMetrics.spikeRenderTime = duration;
    }
    performanceMetrics.lastUpdate = Date.now();

    console.log(`⏱️  ${name}: ${duration.toFixed(2)}ms`);
}
