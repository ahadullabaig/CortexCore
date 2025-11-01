/**
 * CortexCore - Demo JavaScript
 * ==============================================
 *
 * Owner: CS4 / Deployment Engineer
 * Phase: Days 3-30
 */

// ============================================
// Global State
// ============================================

let currentSignal = null;
let currentSpikes = null;

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🧠 CortexCore Demo initialized');

    // Check system health
    checkHealth();

    // Setup event listeners
    setupEventListeners();

    // Initialize empty plots
    initializePlots();
});

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');

    generateBtn.addEventListener('click', generateSample);
    predictBtn.addEventListener('click', runPrediction);
}

// ============================================
// API Calls
// ============================================

async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        // Update status badges
        const modelStatus = document.getElementById('model-status');
        const deviceStatus = document.getElementById('device-status');

        if (data.model.loaded) {
            modelStatus.textContent = 'Loaded';
            modelStatus.className = 'status-badge success';

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
        }

        deviceStatus.textContent = data.device.toUpperCase();
        deviceStatus.className = data.device === 'cuda' ? 'status-badge success' : 'status-badge info';

        console.log('✅ Health check complete:', data);
    } catch (error) {
        console.error('❌ Health check failed:', error);
        const modelStatus = document.getElementById('model-status');
        modelStatus.textContent = 'Error';
        modelStatus.className = 'status-badge danger';
    }
}

async function generateSample() {
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');
    const condition = document.getElementById('condition-select').value;

    try {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';

        const response = await fetch('/api/generate_sample', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                condition: condition,
                duration: 10,
                sampling_rate: 250
            }),
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        currentSignal = data.signal;

        // Plot ECG
        plotECG(data.signal, data.condition);

        // Generate and plot spikes
        await generateSpikes(data.signal);

        // Enable prediction
        predictBtn.disabled = false;

        console.log('✅ Sample generated:', data);

    } catch (error) {
        console.error('❌ Generation failed:', error);
        alert('Failed to generate sample: ' + error.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate ECG Sample';
    }
}

async function generateSpikes(signal) {
    try {
        const response = await fetch('/api/visualize_spikes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                signal: signal
            }),
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        currentSpikes = data;
        plotSpikes(data);

        console.log('✅ Spikes generated:', data);

    } catch (error) {
        console.error('❌ Spike generation failed:', error);
    }
}

async function runPrediction() {
    if (!currentSignal) {
        alert('Please generate a sample first');
        return;
    }

    const predictBtn = document.getElementById('predict-btn');

    try {
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                signal: currentSignal,
                encode: true
            }),
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);

        console.log('✅ Prediction complete:', data);

    } catch (error) {
        console.error('❌ Prediction failed:', error);
        alert('Prediction failed: ' + error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Run Prediction';
    }
}

// ============================================
// Visualization
// ============================================

function initializePlots() {
    // Initialize empty ECG plot
    const ecgLayout = {
        title: 'No signal generated yet',
        xaxis: { title: 'Sample' },
        yaxis: { title: 'Amplitude' },
        margin: { t: 40, r: 20, b: 40, l: 50 }
    };
    Plotly.newPlot('ecg-plot', [], ecgLayout);

    // Initialize empty spike plot
    const spikeLayout = {
        title: 'No spikes generated yet',
        xaxis: { title: 'Time Step' },
        yaxis: { title: 'Neuron Index' },
        margin: { t: 40, r: 20, b: 40, l: 50 }
    };
    Plotly.newPlot('spike-plot', [], spikeLayout);
}

function plotECG(signal, condition) {
    const trace = {
        y: signal,
        type: 'scatter',
        mode: 'lines',
        name: condition,
        line: {
            color: condition === 'normal' ? '#4caf50' : '#f44336',
            width: 2
        }
    };

    const layout = {
        title: `ECG Signal - ${condition.charAt(0).toUpperCase() + condition.slice(1)}`,
        xaxis: {
            title: 'Sample',
            gridcolor: '#e0e0e0'
        },
        yaxis: {
            title: 'Amplitude (normalized)',
            gridcolor: '#e0e0e0'
        },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
        margin: { t: 40, r: 20, b: 40, l: 50 }
    };

    Plotly.newPlot('ecg-plot', [trace], layout);
}

function plotSpikes(spikeData) {
    const trace = {
        x: spikeData.spike_times,
        y: spikeData.neuron_ids,
        mode: 'markers',
        type: 'scatter',
        name: 'Spikes',
        marker: {
            color: '#667eea',
            size: 5,
            symbol: 'line-ns-open',
            line: {
                width: 2
            }
        }
    };

    const layout = {
        title: `Spike Raster Plot - ${spikeData.spike_times.length} total spikes`,
        xaxis: {
            title: 'Time Step',
            range: [0, spikeData.num_steps],
            gridcolor: '#e0e0e0'
        },
        yaxis: {
            title: 'Neuron Index',
            range: [-1, spikeData.num_neurons],
            gridcolor: '#e0e0e0'
        },
        plot_bgcolor: '#fafafa',
        paper_bgcolor: '#ffffff',
        margin: { t: 40, r: 20, b: 40, l: 50 }
    };

    Plotly.newPlot('spike-plot', [trace], layout);
}

// ============================================
// Results Display
// ============================================

function displayResults(results) {
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

    // Update probability bars
    if (results.probabilities && results.probabilities.length >= 2) {
        const probNormal = results.probabilities[0] * 100;
        const probArrhythmia = results.probabilities[1] * 100;

        document.getElementById('prob-normal').style.width = `${probNormal}%`;
        document.getElementById('prob-normal-text').textContent = `${probNormal.toFixed(1)}%`;

        document.getElementById('prob-arrhythmia').style.width = `${probArrhythmia}%`;
        document.getElementById('prob-arrhythmia-text').textContent = `${probArrhythmia.toFixed(1)}%`;
    }

    // Show warning if using mock data
    if (results.warning) {
        console.warn('⚠️', results.warning);
    }

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ============================================
// Utility Functions
// ============================================

function formatTime(ms) {
    if (ms < 1) {
        return `${(ms * 1000).toFixed(2)} µs`;
    } else if (ms < 1000) {
        return `${ms.toFixed(2)} ms`;
    } else {
        return `${(ms / 1000).toFixed(2)} s`;
    }
}

// ============================================
// Console Styling
// ============================================

console.log('%c🧠 CortexCore Demo', 'font-size: 20px; font-weight: bold; color: #667eea;');
console.log('%cBrain-inspired computing for medical diagnosis', 'font-size: 14px; color: #666;');
console.log('');
console.log('📊 Features:');
console.log('  • Real-time ECG generation');
console.log('  • Spike-based encoding');
console.log('  • Energy-efficient inference');
console.log('  • Clinical-grade predictions');
console.log('');
