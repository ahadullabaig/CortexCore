/**
 * Performance Monitoring (Debug Mode)
 * FPS counter and memory usage display
 */

import { performanceMetrics } from '../utils/performance.js';

/**
 * Setup performance monitoring overlay
 * Only activates in debug mode (?debug in URL)
 */
export function setupPerformanceMonitoring() {
    let lastTime = performance.now();
    let frames = 0;
    let fps = 60;

    // Create FPS counter display
    const fpsDisplay = document.createElement('div');
    fpsDisplay.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.8);
        color: #00d9ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        padding: 8px 12px;
        border-radius: 4px;
        border: 1px solid #00d9ff;
        z-index: 10000;
        pointer-events: none;
    `;
    fpsDisplay.id = 'fps-counter';
    document.body.appendChild(fpsDisplay);

    function updateFPS() {
        const currentTime = performance.now();
        frames++;

        if (currentTime >= lastTime + 1000) {
            fps = Math.round((frames * 1000) / (currentTime - lastTime));
            frames = 0;
            lastTime = currentTime;

            // Update display with FPS and memory
            const memoryInfo = performance.memory ?
                ` | Heap: ${(performance.memory.usedJSHeapSize / 1048576).toFixed(1)}MB` :
                '';

            // Include visualization metrics if available
            const vizMetrics = (performanceMetrics.ecgRenderTime > 0 || performanceMetrics.spikeRenderTime > 0) ?
                `\nECG: ${performanceMetrics.ecgRenderTime.toFixed(1)}ms | Spikes: ${performanceMetrics.spikeRenderTime.toFixed(1)}ms | Encoding: ${performanceMetrics.spikeEncodingTime.toFixed(1)}ms` :
                '';

            fpsDisplay.textContent = `FPS: ${fps}${memoryInfo}${vizMetrics}`;
            fpsDisplay.style.borderColor = fps < 30 ? '#ef4444' : fps < 50 ? '#f59e0b' : '#00d9ff';
        }

        requestAnimationFrame(updateFPS);
    }

    requestAnimationFrame(updateFPS);
    console.log('📊 Performance monitoring enabled (debug mode)');
}
