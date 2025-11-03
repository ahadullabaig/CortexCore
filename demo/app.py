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
    from src.inference import load_model, predict
    from src.data import rate_encode
    from src.utils import get_device
    from src.model import SimpleSNN
except ImportError as e:
    print(f"⚠️  Warning: src modules not fully implemented yet: {e}")

# ============================================
# App Configuration
# ============================================

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Load configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pt')
DEVICE = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

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
            # Load SimpleSNN model
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

            # Get model config from checkpoint
            config = checkpoint.get('config', {})
            input_size = config.get('input_size', 2500)
            hidden_size = config.get('hidden_size', 128)

            # Create model instance
            model = SimpleSNN(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=2
            )

            # Load weights using the inference module
            model = load_model(MODEL_PATH, model, device=DEVICE)

            model_info = {
                'loaded': True,
                'path': MODEL_PATH,
                'device': DEVICE,
                'val_acc': checkpoint.get('val_acc', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'input_size': input_size,
                'hidden_size': hidden_size,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            print(f"✅ Model loaded from {MODEL_PATH}")
            print(f"   Validation accuracy: {model_info['val_acc']}")
            print(f"   Parameters: {model_info['parameters']:,}")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
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
    return jsonify({
        'status': 'healthy',
        'model': model_info,
        'device': DEVICE,
        'timestamp': time.time()
    })


# ============================================
# Routes - API
# ============================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Prediction API endpoint

    Expected JSON:
    {
        "signal": [array of values],
        "num_steps": 100 (optional)
    }

    Returns:
    {
        "prediction": class_id,
        "confidence": confidence_score,
        "probabilities": [prob1, prob2, ...],
        "class_name": "Normal" or "Arrhythmia",
        "inference_time_ms": time_in_ms,
        "spike_count": number_of_spikes
    }
    """
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'details': model_info.get('error', 'Unknown error')
            }), 503

        data = request.json

        if 'signal' not in data:
            return jsonify({'error': 'Missing signal data'}), 400

        signal = np.array(data['signal'])
        num_steps = data.get('num_steps', 100)

        # Ensure signal is the correct length (2500 samples)
        if len(signal) != 2500:
            return jsonify({
                'error': f'Signal must be 2500 samples, got {len(signal)}'
            }), 400

        # Make prediction using trained SNN
        result = predict(
            model,
            signal,
            device=DEVICE,
            return_confidence=True,
            num_steps=num_steps
        )

        return jsonify(result)

    except Exception as e:
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
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'details': model_info.get('error', 'Unknown error')
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

        # Convert to tensor
        signal_tensor = torch.FloatTensor(signal)

        # Normalize signal to [0, 1] range
        signal_min = signal_tensor.min()
        signal_max = signal_tensor.max()
        if signal_max > signal_min:
            signal_normalized = (signal_tensor - signal_min) / (signal_max - signal_min)
        else:
            signal_normalized = torch.zeros_like(signal_tensor)

        # Generate spike representation: replicate signal across time steps
        # Shape: [num_steps, signal_length]
        spike_array = signal_normalized.unsqueeze(0).repeat(num_steps, 1).numpy()

        # Apply threshold to convert to binary spikes (for visualization)
        # Use gain as threshold sensitivity
        spike_threshold = 1.0 / gain
        spike_array = (spike_array > spike_threshold).astype(float)

        # Sample neurons for visualization
        num_neurons_to_show = min(128, spike_array.shape[1])

        spike_times = []
        neuron_ids = []

        # Extract spike times for each neuron
        for neuron_idx in range(num_neurons_to_show):
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
