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

    // Phase 3: Initialize animation systems
    initializeAnimationSystems();

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
        // Measure spike encoding time
        const data = await measureVisualizationTime('spike-encoding', async () => {
            const response = await fetch('/api/visualize_spikes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    signal: signal
                }),
            });

            return await response.json();
        });

        if (data.error) {
            throw new Error(data.error);
        }

        currentSpikes = data;

        // Update spike statistics with animated counters
        const totalSpikes = data.spike_times.length;
        const duration = 10.0; // ECG signal duration is always 10 seconds
        const firingRate = totalSpikes / duration;
        const sparsity = (totalSpikes / (data.num_neurons * data.num_steps)) * 100;

        // Animate all three counters with staggered timing
        animateCountUp(document.getElementById('total-spikes'), 0, totalSpikes, 800);
        animateCountUp(document.getElementById('firing-rate'), 0, firingRate, 800, { suffix: ' Hz', decimals: 1 });
        animateCountUp(document.getElementById('sparsity'), 0, sparsity, 800, { suffix: '%', decimals: 1 });

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
    // Initialize empty ECG plot with dark theme
    const ecgLayout = {
        title: {
            text: 'No signal generated yet',
            font: {
                family: 'JetBrains Mono, monospace',
                color: '#b4bcd0'
            }
        },
        xaxis: {
            title: {
                text: 'Sample',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            color: '#b4bcd0'
        },
        yaxis: {
            title: {
                text: 'Amplitude',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            color: '#b4bcd0'
        },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        margin: { t: 60, r: 20, b: 60, l: 70 },
        font: {
            family: 'JetBrains Mono, monospace',
            color: '#b4bcd0'
        }
    };
    Plotly.newPlot('ecg-plot', [], ecgLayout);

    // Initialize empty spike plot with dark theme
    const spikeLayout = {
        title: {
            text: 'No spikes generated yet',
            font: {
                family: 'JetBrains Mono, monospace',
                color: '#b4bcd0'
            }
        },
        xaxis: {
            title: {
                text: 'Time Step',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            color: '#b4bcd0'
        },
        yaxis: {
            title: {
                text: 'Neuron Index',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            color: '#b4bcd0'
        },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        margin: { t: 60, r: 20, b: 60, l: 70 },
        font: {
            family: 'JetBrains Mono, monospace',
            color: '#b4bcd0'
        }
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
        title: {
            text: `ECG Signal - ${condition.charAt(0).toUpperCase() + condition.slice(1)}`,
            font: {
                family: 'JetBrains Mono, monospace',
                color: '#b4bcd0'
            }
        },
        xaxis: {
            title: {
                text: 'Sample',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            zerolinecolor: 'rgba(180, 188, 208, 0.2)',
            color: '#b4bcd0'
        },
        yaxis: {
            title: {
                text: 'Amplitude (normalized)',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            zerolinecolor: 'rgba(180, 188, 208, 0.2)',
            color: '#b4bcd0'
        },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        margin: { t: 60, r: 20, b: 60, l: 70 },
        font: {
            family: 'JetBrains Mono, monospace',
            color: '#b4bcd0'
        }
    };

    Plotly.newPlot('ecg-plot', [trace], layout);

    // Add progressive trace drawing animation for visual impact
    // This simulates the ECG being "drawn" in real-time
    const totalPoints = signal.length;
    const animationDuration = 2000; // 2 seconds total
    const fps = 60; // Target 60 FPS
    const totalFrames = (animationDuration / 1000) * fps;
    const pointsPerFrame = Math.ceil(totalPoints / totalFrames);
    const frameDelay = 1000 / fps;

    let currentPoint = 0;

    function drawNextSegment() {
        currentPoint += pointsPerFrame;

        if (currentPoint >= totalPoints) {
            // Final update with complete data
            Plotly.restyle('ecg-plot', {
                y: [signal]
            }, [0]);
            return;
        }

        // Update with partial data (progressive reveal)
        const partialSignal = signal.slice(0, currentPoint);
        Plotly.restyle('ecg-plot', {
            y: [partialSignal]
        }, [0]);

        // Continue animation
        setTimeout(drawNextSegment, frameDelay);
    }

    // Start drawing animation after brief delay
    setTimeout(drawNextSegment, 100);
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
        title: {
            text: `Spike Raster Plot - ${spikeData.spike_times.length} total spikes`,
            font: {
                family: 'JetBrains Mono, monospace',
                color: '#b4bcd0'
            }
        },
        xaxis: {
            title: {
                text: 'Time Step',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            range: [0, spikeData.num_steps],
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            zerolinecolor: 'rgba(180, 188, 208, 0.2)',
            color: '#b4bcd0'
        },
        yaxis: {
            title: {
                text: 'Neuron Index',
                font: { family: 'JetBrains Mono, monospace', color: '#b4bcd0' }
            },
            range: [-1, spikeData.num_neurons],
            gridcolor: 'rgba(180, 188, 208, 0.1)',
            zerolinecolor: 'rgba(180, 188, 208, 0.2)',
            color: '#b4bcd0'
        },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        margin: { t: 60, r: 20, b: 60, l: 70 },
        font: {
            family: 'JetBrains Mono, monospace',
            color: '#b4bcd0'
        }
    };

    Plotly.newPlot('spike-plot', [trace], layout);

    // Add progressive reveal animation for temporal dynamics visualization
    // Group spikes by time step
    const spikesByTime = {};
    for (let i = 0; i < spikeData.spike_times.length; i++) {
        const timeStep = spikeData.spike_times[i];
        if (!spikesByTime[timeStep]) {
            spikesByTime[timeStep] = { times: [], neurons: [] };
        }
        spikesByTime[timeStep].times.push(timeStep);
        spikesByTime[timeStep].neurons.push(spikeData.neuron_ids[i]);
    }

    const timeSteps = Object.keys(spikesByTime).map(Number).sort((a, b) => a - b);

    // Progressive reveal: gradually show spikes based on their temporal order
    // This creates a "firing" effect that shows neural dynamics
    const animationDuration = 1500; // 1.5 seconds total
    const numBatches = Math.min(timeSteps.length, 40); // Max 40 frames for smooth animation
    const batchSize = Math.ceil(timeSteps.length / numBatches);
    const delayPerBatch = animationDuration / numBatches;

    let accumulatedX = [];
    let accumulatedY = [];
    let currentBatch = 0;

    function revealNextBatch() {
        if (currentBatch >= numBatches) return;

        // Add spikes from current batch (time-ordered)
        const startIdx = currentBatch * batchSize;
        const endIdx = Math.min(startIdx + batchSize, timeSteps.length);

        for (let i = startIdx; i < endIdx; i++) {
            const timeStep = timeSteps[i];
            const spikes = spikesByTime[timeStep];
            accumulatedX.push(...spikes.times);
            accumulatedY.push(...spikes.neurons);
        }

        // Update plot with accumulated spikes (smooth progressive reveal)
        Plotly.restyle('spike-plot', {
            x: [accumulatedX],
            y: [accumulatedY]
        }, [0]);

        currentBatch++;

        if (currentBatch < numBatches) {
            setTimeout(revealNextBatch, delayPerBatch);
        }
    }

    // Start progressive reveal after brief delay
    setTimeout(revealNextBatch, 200);
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


// ============================================
// PHASE 3: ANIMATION SYSTEMS
// ============================================

// ============================================
// Animation System Initialization
// ============================================

function initializeAnimationSystems() {
    console.log('🎬 Initializing Phase 3 Animation Systems...');

    // 1. Detect device capabilities and set optimization class
    detectDeviceCapabilities();

    // 2. Initialize Intersection Observer for scroll-triggered animations
    initializeScrollAnimations();

    // 3. Setup button ripple effects
    setupButtonRipples();

    // 4. Create floating particles (if not low-end device)
    if (!document.body.classList.contains('low-end-device')) {
        createFloatingParticles(25); // 25 particles
    }

    // 5. Create grid scanner
    if (!document.body.classList.contains('low-end-device')) {
        createGridScanner();
    }

    // 6. Setup performance monitoring (dev mode)
    if (window.location.search.includes('debug')) {
        setupPerformanceMonitoring();
    }

    // 7. Pause animations when tab is inactive (battery saving)
    setupVisibilityHandler();

    console.log('✅ Animation systems initialized');
}


// ============================================
// Device Capability Detection
// ============================================

function detectDeviceCapabilities() {
    const body = document.body;

    // Check if low-end device (limited memory, slow CPU, or mobile)
    const isLowEnd = (
        navigator.hardwareConcurrency <= 2 || // 2 or fewer CPU cores
        navigator.deviceMemory <= 4 || // 4GB or less RAM
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    );

    // Check battery status if available
    if (navigator.getBattery) {
        navigator.getBattery().then(battery => {
            // If battery is low (<20%) and not charging, disable heavy animations
            if (battery.level < 0.2 && !battery.charging) {
                body.classList.add('low-end-device');
                console.log('⚡ Low battery detected - disabling heavy animations');
            }
        });
    }

    if (isLowEnd) {
        body.classList.add('low-end-device');
        console.log('📱 Low-end device detected - optimizing animations');
    } else {
        console.log('💪 High-performance device detected - full animations enabled');
    }
}


// ============================================
// Scroll-Triggered Animations (Intersection Observer)
// ============================================

function initializeScrollAnimations() {
    // Create intersection observer
    const observerOptions = {
        root: null, // viewport
        rootMargin: '0px',
        threshold: 0.1 // Trigger when 10% of element is visible
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Add 'animated' class to trigger animation
                entry.target.classList.add('animated');

                // Optional: Unobserve after animation (one-time animation)
                // observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all elements with .animate-on-scroll class
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach(el => observer.observe(el));

    console.log(`👁️ Observing ${animatedElements.length} elements for scroll animations`);
}


// ============================================
// Button Ripple Effects
// ============================================

function setupButtonRipples() {
    const buttons = document.querySelectorAll('.btn');

    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Don't add ripple if button is disabled
            if (this.disabled) return;

            // Create ripple element
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');

            // Get click position relative to button
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Position ripple at click location
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';

            // Add ripple to button
            this.appendChild(ripple);

            // Remove ripple after animation completes (600ms)
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    console.log(`💫 Button ripple effects enabled for ${buttons.length} buttons`);
}


// ============================================
// Floating Particles System
// ============================================

function createFloatingParticles(count) {
    // Create container if it doesn't exist
    let container = document.querySelector('.neural-particles');

    if (!container) {
        container = document.createElement('div');
        container.classList.add('neural-particles');
        document.body.appendChild(container);
    }

    // Clear existing particles
    container.innerHTML = '';

    // Create particles
    for (let i = 0; i < count; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');

        // Randomize size
        const sizeRandom = Math.random();
        if (sizeRandom < 0.3) {
            particle.classList.add('small');
        } else if (sizeRandom > 0.7) {
            particle.classList.add('large');
        }

        // Random starting position
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';

        // Random animation duration (10-30 seconds)
        particle.style.animationDuration = (10 + Math.random() * 20) + 's';

        // Random animation delay for stagger effect
        particle.style.animationDelay = (Math.random() * 5) + 's';

        container.appendChild(particle);
    }

    console.log(`✨ Created ${count} floating particles`);
}


// ============================================
// Grid Scanner Effect
// ============================================

function createGridScanner() {
    // Create scanner if it doesn't exist
    if (!document.querySelector('.grid-scanner')) {
        const scanner = document.createElement('div');
        scanner.classList.add('grid-scanner');
        document.body.appendChild(scanner);
        console.log('📡 Grid scanner effect enabled');
    }
}


// ============================================
// Number Counter Animations
// ============================================

function animateCountUp(element, start, end, duration = 1000, options = {}) {
    const { suffix = '', decimals = 0 } = options;
    const startTime = performance.now();
    const range = end - start;

    function updateCount(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out cubic)
        const easeProgress = 1 - Math.pow(1 - progress, 3);

        const current = start + range * easeProgress;
        const formattedValue = decimals > 0 ? current.toFixed(decimals) : Math.floor(current).toLocaleString();
        element.textContent = formattedValue + suffix;

        if (progress < 1) {
            requestAnimationFrame(updateCount);
        } else {
            const finalValue = decimals > 0 ? end.toFixed(decimals) : end.toLocaleString();
            element.textContent = finalValue + suffix;
            element.classList.add('counter-animated');
        }
    }

    requestAnimationFrame(updateCount);
}


// ============================================
// Progress Bar Smooth Animation
// ============================================

function animateProgressBar(element, targetWidth, duration = 800) {
    const startTime = performance.now();
    const startWidth = parseFloat(element.style.width) || 0;
    const widthChange = targetWidth - startWidth;

    function updateWidth(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeProgress = 1 - Math.pow(1 - progress, 2);

        const currentWidth = startWidth + widthChange * easeProgress;
        element.style.width = currentWidth + '%';

        if (progress < 1) {
            requestAnimationFrame(updateWidth);
        }
    }

    requestAnimationFrame(updateWidth);
}


// ============================================
// Loading Spinner Management
// ============================================

function showLoadingSpinner(container, size = 'normal') {
    const spinner = document.createElement('div');
    spinner.classList.add('loading-spinner');
    if (size === 'small') {
        spinner.classList.add('small');
    }
    spinner.id = 'active-spinner';

    if (typeof container === 'string') {
        document.getElementById(container).appendChild(spinner);
    } else {
        container.appendChild(spinner);
    }

    return spinner;
}

function hideLoadingSpinner() {
    const spinner = document.getElementById('active-spinner');
    if (spinner) {
        spinner.remove();
    }
}


// ============================================
// Performance Monitoring (Debug Mode)
// ============================================

function setupPerformanceMonitoring() {
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

// Global performance metrics storage
const performanceMetrics = {
    ecgRenderTime: 0,
    spikeRenderTime: 0,
    spikeEncodingTime: 0,
    lastUpdate: Date.now()
};

// Helper function to measure and display visualization performance
function measureVisualizationTime(name, callback) {
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

// Update stored metrics (display updated by FPS loop in debug mode)
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


// ============================================
// Visibility Handler (Pause animations when tab inactive)
// ============================================

function setupVisibilityHandler() {
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            // Pause heavy animations
            document.body.style.animationPlayState = 'paused';
            console.log('⏸️ Animations paused (tab inactive)');
        } else {
            // Resume animations
            document.body.style.animationPlayState = 'running';
            console.log('▶️ Animations resumed (tab active)');
        }
    });
}


// ============================================
// Enhanced Plotly Animations
// ============================================

// Override plot functions to add custom animations
const originalPlotECG = plotECG;
const originalPlotSpikes = plotSpikes;

// Enhanced ECG plotting with animation
window.plotECG = function(signal, condition) {
    // Measure render time and call original function
    measureVisualizationTime('ecg-render', () => {
        originalPlotECG(signal, condition);
    });

    // Add entrance animation to plot container
    const plotContainer = document.getElementById('ecg-plot');
    if (plotContainer) {
        plotContainer.style.animation = 'fadeInUp 0.6s ease-out';
    }
};

// Enhanced spike plotting with animation
window.plotSpikes = function(spikeData) {
    // Measure render time and call original function
    measureVisualizationTime('spike-render', () => {
        originalPlotSpikes(spikeData);
    });

    // Add entrance animation to plot container
    const plotContainer = document.getElementById('spike-plot');
    if (plotContainer) {
        plotContainer.style.animation = 'scaleIn 0.6s ease-out';
    }

    // Animate spike statistics with counter effect (now handled in generateSpikes)
    // Counter animations are already applied in generateSpikes function
};
