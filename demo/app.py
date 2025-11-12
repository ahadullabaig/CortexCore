"""
Demo Application - Flask Server
================================

Owner: CS4 / Deployment Engineer

Responsibilities:
- Web interface for model demonstration
- Real-time prediction API
- Visualization endpoints
- Performance monitoring

Phase: Days 3-30
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.inference import load_model, predict, ensemble_predict
    from src.data import rate_encode
    from src.utils import get_device
    from src.model import SimpleSNN, HybridSTDP_SNN, DeepSNN, WiderSNN
    from src.stdp import STDPConfig
except ImportError as e:
    print(f"⚠️  Warning: src modules not fully implemented yet: {e}")

# ============================================
# App Configuration
# ============================================

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Load configuration
# Default to MVP model path first, fall back to STDP model if available
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pt')
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Classification threshold for high-sensitivity arrhythmia detection
# Lower threshold = higher sensitivity (fewer missed arrhythmias)
# Default 0.40 targets 95% sensitivity based on ROC analysis
SENSITIVITY_THRESHOLD = float(os.getenv('SENSITIVITY_THRESHOLD', '0.40'))

# Global model variable
model = None
model_info = {}

# ============================================
# Model Loading
# ============================================

def init_model():
    """Initialize and load model"""
    global model, model_info

    if Path(MODEL_PATH).exists():
        try:
            # Load checkpoint (use weights_only=False for backward compatibility)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

            # Get model config from checkpoint
            config = checkpoint.get('config', {})
            input_size = config.get('input_size', 2500)
            output_size = config.get('output_size', 2)

            # Detect architecture type from config
            architecture = config.get('architecture', None)

            # If no architecture specified, try to detect from other indicators
            if architecture is None:
                has_stdp_stats = 'stdp_statistics' in checkpoint
                has_stdp_path = 'stdp' in MODEL_PATH.lower()
                is_stdp_model = has_stdp_stats or has_stdp_path

                if is_stdp_model:
                    architecture = 'HybridSTDP_SNN'
                else:
                    architecture = 'SimpleSNN'

            model_type = architecture
            print(f"🔍 Model Detection:")
            print(f"   Architecture: {architecture}")
            print(f"   Config: {config}")

            # Create model instance based on architecture
            if architecture == 'DeepSNN':
                print(f"🧠 Loading DeepSNN model...")
                hidden_sizes = config.get('hidden_sizes', [256, 128])
                dropout = config.get('dropout', 0.3)
                beta = config.get('beta', 0.9)

                model = DeepSNN(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    beta=beta,
                    dropout=dropout
                )
            elif architecture == 'WiderSNN':
                print(f"🧠 Loading WiderSNN model...")
                hidden_size = config.get('hidden_size', 256)
                dropout = config.get('dropout', 0.2)
                beta = config.get('beta', 0.9)

                model = WiderSNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    beta=beta,
                    dropout=dropout
                )
            elif architecture == 'HybridSTDP_SNN':
                print(f"🧠 Loading STDP model...")
                hidden_size = config.get('hidden_size', 128)
                # Create STDP config
                stdp_config = STDPConfig(
                    use_homeostasis=True,
                    target_rate=10.0,
                    use_multiscale=True,
                    tau_fast=10.0,
                    tau_slow=100.0,
                    alpha_initial=0.8,
                    alpha_final=0.3
                )
                model = HybridSTDP_SNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    config=stdp_config
                )
            else:  # Default to SimpleSNN
                print(f"🧠 Loading SimpleSNN model...")
                hidden_size = config.get('hidden_size', 128)
                model = SimpleSNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size
                )

            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()

            # Get validation accuracy from checkpoint
            if 'metrics' in checkpoint:
                val_acc = checkpoint['metrics'].get('val_accuracy', 'N/A')
            else:
                val_acc = checkpoint.get('val_acc', checkpoint.get('val_accuracy', 'N/A'))

            # Get additional info for model_info
            if architecture == 'DeepSNN':
                architecture_info = f"DeepSNN ({config.get('hidden_sizes', [256, 128])})"
            elif architecture == 'WiderSNN':
                architecture_info = f"WiderSNN ({config.get('hidden_size', 256)})"
            else:
                architecture_info = f"{architecture} ({config.get('hidden_size', 128)})"

            model_info = {
                'loaded': True,
                'model_type': model_type,
                'architecture': architecture,
                'architecture_info': architecture_info,
                'path': MODEL_PATH,
                'device': DEVICE,
                'val_acc': val_acc,
                'epoch': checkpoint.get('epoch', 'N/A'),
                'config': config,
                'parameters': sum(p.numel() for p in model.parameters()),
                'stdp_enabled': architecture == 'HybridSTDP_SNN'
            }
            print(f"✅ Model loaded from {MODEL_PATH}")
            print(f"   Type: {model_type}")
            print(f"   Validation accuracy: {model_info['val_acc']}")
            print(f"   Parameters: {model_info['parameters']:,}")
            print(f"   STDP features: {model_info['stdp_enabled']}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            import traceback
            traceback.print_exc()
            model_info = {'loaded': False, 'error': str(e)}
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        model_info = {'loaded': False, 'error': 'Model file not found'}

# ============================================
# Routes - Frontend
# ============================================

@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    # Calculate device memory if CUDA available
    device_memory = 0
    if torch.cuda.is_available():
        device_memory = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9  # Convert to GB

    return jsonify({
        'status': 'healthy',
        'model': model_info,
        'device': DEVICE,
        'device_memory': device_memory,
        'timestamp': time.time()
    })


# ============================================
# Routes - API
# ============================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Prediction API endpoint with optional ensemble averaging

    Expected JSON:
    {
        "signal": [array of values],
        "num_steps": 100 (optional),
        "ensemble_size": 5 (optional, enables ensemble averaging),
        "use_seed": false (optional, for reproducible predictions)
    }

    Returns:
    {
        "prediction": class_id,
        "confidence": confidence_score,
        "probabilities": [prob1, prob2, ...],
        "class_name": "Normal" or "Arrhythmia",
        "inference_time_ms": time_in_ms,
        "spike_count": number_of_spikes,

        # If ensemble_size > 1, additional fields:
        "confidence_std": uncertainty_measure,
        "confidence_ci_95": [lower, upper],
        "agreement_rate": percentage_of_runs_agreeing,
        "ensemble_size": number_of_runs,
        "is_ensemble": true/false
    }
    """
    try:
        # Strengthen model validation
        if model is None or not model_info.get('loaded', False):
            return jsonify({
                'error': 'Model not loaded',
                'details': model_info.get('error', 'Unknown error'),
                'suggestion': 'Please check server logs and retrain model if necessary'
            }), 503

        data = request.json

        if 'signal' not in data:
            return jsonify({'error': 'Missing signal data'}), 400

        signal = np.array(data['signal'])
        num_steps = data.get('num_steps', 100)
        ensemble_size = data.get('ensemble_size', 1)  # Default: single prediction
        use_seed = data.get('use_seed', False)
        seed = 42 if use_seed else None

        # Ensure signal is the correct length (2500 samples)
        if len(signal) != 2500:
            return jsonify({
                'error': f'Signal must be 2500 samples, got {len(signal)}'
            }), 400

        # Validate ensemble_size
        if ensemble_size < 1 or ensemble_size > 10:
            return jsonify({
                'error': f'ensemble_size must be between 1 and 10, got {ensemble_size}'
            }), 400

        # Make prediction (ensemble or single) with calibrated threshold
        if ensemble_size > 1:
            result = ensemble_predict(
                model,
                signal,
                ensemble_size=ensemble_size,
                device=DEVICE,
                num_steps=num_steps,
                base_seed=seed,
                return_confidence=True,
                sensitivity_threshold=SENSITIVITY_THRESHOLD
            )
            result['is_ensemble'] = True
        else:
            result = predict(
                model,
                signal,
                device=DEVICE,
                return_confidence=True,
                num_steps=num_steps,
                seed=seed,
                sensitivity_threshold=SENSITIVITY_THRESHOLD
            )
            result['is_ensemble'] = False
            result['ensemble_size'] = 1

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_sample', methods=['POST'])
def api_generate_sample():
    """
    Generate synthetic sample for testing

    TODO: Day 3-4
        - Use src.data.generate_synthetic_ecg
        - Return signal data for visualization
    """
    try:
        data = request.json
        condition = data.get('condition', 'normal')

        # TODO: Use actual data generation
        # Mock ECG data for now
        import neurokit2 as nk

        duration = data.get('duration', 10)
        sampling_rate = data.get('sampling_rate', 250)

        if condition == 'normal':
            ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70)
        else:
            ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=120)

        return jsonify({
            'signal': ecg.tolist(),
            'condition': condition,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'length': len(ecg)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize_spikes', methods=['POST'])
def api_visualize_spikes():
    """
    Generate spike visualization data from signal

    Expected JSON:
    {
        "signal": [array of values],
        "num_steps": 100 (optional),
        "gain": 10.0 (optional)
    }

    Returns spike raster data for visualization
    """
    try:
        # Strengthen model validation
        if model is None or not model_info.get('loaded', False):
            return jsonify({
                'error': 'Model not loaded',
                'details': model_info.get('error', 'Unknown error'),
                'suggestion': 'Please check server logs and retrain model if necessary'
            }), 503

        data = request.json
        signal = np.array(data['signal'])
        num_steps = data.get('num_steps', 100)
        gain = data.get('gain', 10.0)

        # Ensure signal is correct length
        if len(signal) != 2500:
            return jsonify({
                'error': f'Signal must be 2500 samples, got {len(signal)}'
            }), 400

        # Create 2D spike encoding for visualization
        # Normalize signal to [0, 1]
        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)

        # Replicate normalized signal across time steps and apply Poisson encoding
        # Shape: [num_steps, signal_length]
        spike_array = np.random.rand(num_steps, len(signal)) < (signal_norm * gain / 100.0)
        spike_array = spike_array.astype(np.float32)

        # Sample neurons for visualization (display subset for performance)
        num_neurons_to_show = min(128, spike_array.shape[1])
        stride = max(1, spike_array.shape[1] // num_neurons_to_show)

        spike_times = []
        neuron_ids = []

        # Extract spike times for sampled neurons
        for idx, neuron_idx in enumerate(range(0, spike_array.shape[1], stride)):
            if idx >= num_neurons_to_show:
                break
            neuron_spikes = spike_array[:, neuron_idx]
            time_indices = np.where(neuron_spikes > 0)[0]
            spike_times.extend(time_indices.tolist())
            neuron_ids.extend([neuron_idx] * len(time_indices))

        # Calculate spike statistics
        total_spikes = len(spike_times)
        spike_rate = total_spikes / (num_steps * num_neurons_to_show) if num_neurons_to_show > 0 else 0

        return jsonify({
            'spike_times': spike_times,
            'neuron_ids': neuron_ids,
            'num_steps': num_steps,
            'num_neurons': num_neurons_to_show,
            'total_spikes': total_spikes,
            'spike_rate': float(spike_rate),
            'gain': gain
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# Routes - Monitoring
# ============================================

@app.route('/api/metrics')
def api_metrics():
    """
    Get system metrics

    TODO: Day 14+
        - Add GPU memory usage
        - Add inference statistics
        - Add model performance metrics
    """
    metrics = {
        'model_loaded': model_info.get('loaded', False),
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
        metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1e9

    return jsonify(metrics)


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================
# Main
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 CortexCore - DEMO SERVER")
    print("=" * 60)
    print()

    # Initialize model
    init_model()

    # Configuration
    host = os.getenv('DEMO_HOST', '0.0.0.0')
    port = int(os.getenv('DEMO_PORT', 5000))

    print(f"🌐 Server Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {DEBUG}")
    print(f"   Device: {DEVICE}")
    print()
    print(f"📍 Access demo at: http://localhost:{port}")
    print()
    print("=" * 60)
    print()

    # Run server
    app.run(
        host=host,
        port=port,
        debug=DEBUG,
        threaded=True
    )
