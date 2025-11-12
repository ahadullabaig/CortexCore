/**
 * Health Check API
 * Monitors model and device status
 */

import { AppState } from '../core/state.js';
import { ErrorHandler } from './client.js';

/**
 * Check system health and update status UI
 */
export async function checkHealth() {
    try {
        // Use ErrorHandler for resilient fetching
        const response = await ErrorHandler.fetchWithRetry('/health', {
            timeout: 5000 // 5 second timeout for health checks
        });
        const data = await response.json();

        // Update state
        AppState.setState('model.loaded', data.model.loaded);
        AppState.setState('model.device', data.device);
        if (data.model.val_acc && data.model.val_acc !== 'N/A') {
            AppState.setState('model.accuracy', data.model.val_acc);
        }

        // Update status badges
        const modelStatus = document.getElementById('model-status');
        const deviceStatus = document.getElementById('device-status');
        const modelIndicator = document.getElementById('model-indicator');

        if (data.model.loaded) {
            modelStatus.textContent = 'Loaded';
            modelStatus.className = 'status-badge success';

            // Update model indicator dot to green
            if (modelIndicator) {
                modelIndicator.className = 'label-indicator online';
                modelIndicator.setAttribute('aria-label', 'Model loaded and ready');
            }

            // Update model accuracy if available
            const accElement = document.getElementById('model-accuracy');
            if (data.model.val_acc && data.model.val_acc !== 'N/A') {
                accElement.textContent = `${data.model.val_acc.toFixed(2)}%`;
            } else {
                accElement.textContent = 'N/A';
            }
        } else {
            modelStatus.textContent = 'Not Loaded';
            modelStatus.className = 'status-badge warning';

            // Update model indicator dot to amber/loading
            if (modelIndicator) {
                modelIndicator.className = 'label-indicator loading';
                modelIndicator.setAttribute('aria-label', 'Model not loaded');
            }
        }

        deviceStatus.textContent = data.device.toUpperCase();
        deviceStatus.className = data.device === 'cuda' ? 'status-badge success' : 'status-badge info';

        // Update device memory (VRAM)
        const deviceMemory = document.getElementById('device-memory');
        if (data.device_memory && data.device_memory > 0) {
            deviceMemory.textContent = `${data.device_memory.toFixed(1)} GB VRAM`;
        } else {
            deviceMemory.textContent = 'N/A';
        }

        console.log('✅ Health check complete:', data);
    } catch (error) {
        console.error('❌ Health check failed:', error);
        const modelStatus = document.getElementById('model-status');
        modelStatus.textContent = 'Error';
        modelStatus.className = 'status-badge danger';

        // Don't show toast for health check failures (silent)
        // ErrorHandler already logged the error
    }
}
