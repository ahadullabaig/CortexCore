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
except ImportError:
    print("⚠️  Warning: src modules not fully implemented yet")

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
            # TODO: Load actual SNN model from src.model
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model_info = {
                'loaded': True,
                'path': MODEL_PATH,
                'device': DEVICE,
                'val_acc': checkpoint.get('val_acc', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A')
            }
            print(f"✅ Model loaded from {MODEL_PATH}")
            print(f"   Validation accuracy: {model_info['val_acc']}")
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
        "encode": true/false
    }

    Returns:
    {
        "prediction": class_id,
        "confidence": confidence_score,
        "probabilities": [prob1, prob2, ...],
        "class_name": "Normal" or "Arrhythmia",
        "inference_time_ms": time_in_ms
    }

    TODO: Day 4-5
        - Implement actual prediction
        - Add spike encoding
        - Return spike patterns for visualization
    """
    try:
        data = request.json

        if 'signal' not in data:
            return jsonify({'error': 'Missing signal data'}), 400

        signal = np.array(data['signal'])

        # TODO: Implement actual prediction
        # For now, return mock data
        result = {
            'prediction': int(np.random.randint(0, 2)),
            'confidence': float(np.random.rand()),
            'probabilities': [float(x) for x in np.random.rand(2)],
            'class_name': 'Normal' if np.random.rand() > 0.5 else 'Arrhythmia',
            'inference_time_ms': float(np.random.rand() * 100),
            'warning': 'Using mock predictions - model not implemented yet'
        }

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
    Generate spike visualization data

    TODO: Day 5-6
        - Encode signal to spikes
        - Return spike times for raster plot
    """
    try:
        data = request.json
        signal = np.array(data['signal'])

        # TODO: Use actual spike encoding
        # Mock spike data for now
        num_steps = 100
        num_neurons = 10

        spike_times = []
        neuron_ids = []

        for neuron in range(num_neurons):
            n_spikes = np.random.randint(5, 15)
            times = np.random.choice(num_steps, size=n_spikes, replace=False)
            spike_times.extend(times.tolist())
            neuron_ids.extend([neuron] * n_spikes)

        return jsonify({
            'spike_times': spike_times,
            'neuron_ids': neuron_ids,
            'num_steps': num_steps,
            'num_neurons': num_neurons,
            'warning': 'Using mock spike data - encoding not implemented yet'
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
    print("🧠 NEUROMORPHIC SNN HEALTHCARE - DEMO SERVER")
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
