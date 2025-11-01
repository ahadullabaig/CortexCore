#!/bin/bash

# ============================================
# CortexCore - DEMO LAUNCHER
# ============================================
# Launches Flask demo application
# Usage: bash scripts/04_run_demo.sh
# Access: http://localhost:5000

set -e  # Exit on error

echo "🎬 ============================================"
echo "🎬 DEMO APPLICATION LAUNCHER"
echo "🎬 ============================================"
echo ""

# ==========================================
# Configuration
# ==========================================

# Load from .env if exists
if [ -f ".env" ]; then
    source .env
fi

# Set defaults
export FLASK_APP=${FLASK_APP:-demo/app.py}
export FLASK_ENV=${FLASK_ENV:-development}
export FLASK_DEBUG=${FLASK_DEBUG:-True}
export DEMO_HOST=${DEMO_HOST:-0.0.0.0}
export DEMO_PORT=${DEMO_PORT:-5000}
export MODEL_PATH=${MODEL_PATH:-models/best_model.pt}

echo "📋 Configuration:"
echo "   Flask app: $FLASK_APP"
echo "   Environment: $FLASK_ENV"
echo "   Host: $DEMO_HOST"
echo "   Port: $DEMO_PORT"
echo "   Model: $MODEL_PATH"
echo ""

# ==========================================
# Check Model Availability
# ==========================================

echo "🔍 Checking for trained model..."

if [ ! -f "$MODEL_PATH" ]; then
    echo "   ⚠️  WARNING: Model not found at $MODEL_PATH"
    echo "   The demo will run but predictions may not work"
    echo "   Please train a model first: bash scripts/03_train_mvp_model.sh"
    echo ""
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "   ✅ Model found"
fi

echo ""

# ==========================================
# Check Flask Availability
# ==========================================

echo "🔍 Checking Flask installation..."

python -c "import flask" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ Flask available"
else
    echo "   ❌ ERROR: Flask not installed"
    echo "   Please run: bash scripts/01_setup_environment.sh"
    exit 1
fi

echo ""

# ==========================================
# Check if Port is Available
# ==========================================

echo "🔍 Checking port availability..."

if command -v lsof &> /dev/null; then
    if lsof -Pi :$DEMO_PORT -sTCP:LISTEN -t >/dev/null ; then
        echo "   ⚠️  WARNING: Port $DEMO_PORT is already in use"
        echo "   Another application might be running"
        echo ""
        read -p "   Try to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "   ✅ Port $DEMO_PORT is available"
    fi
else
    echo "   ℹ️  Cannot check port (lsof not available)"
fi

echo ""

# ==========================================
# Launch Demo
# ==========================================

echo "🚀 Launching demo application..."
echo ""
echo "   📍 URL: http://localhost:$DEMO_PORT"
echo "   📍 Network: http://$DEMO_HOST:$DEMO_PORT"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if demo/app.py exists, if not create a minimal one
if [ ! -f "demo/app.py" ]; then
    echo "⚠️  demo/app.py not found, creating minimal version..."
    cat > demo/app.py << 'EOF'
from flask import Flask, render_template, jsonify
import torch
import os

app = Flask(__name__)

# Load model if exists
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pt')
model = None

try:
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        print(f"Model loaded from {MODEL_PATH}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CortexCore</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .status {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 14px;
            }
            .status.success {
                background: #4caf50;
                color: white;
            }
            .status.warning {
                background: #ff9800;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 CortexCore</h1>
            <p>Brain-inspired computing for medical diagnosis</p>
        </div>

        <div class="card">
            <h2>System Status</h2>
            <p>Demo server: <span class="status success">Running</span></p>
            <p>Model status: <span class="status ''' + ('success' if model else 'warning') + '''">
                ''' + ('Loaded' if model else 'Not loaded') + '''
            </span></p>
        </div>

        <div class="card">
            <h2>Next Steps</h2>
            <ol>
                <li>Implement demo UI in <code>demo/templates/index.html</code></li>
                <li>Add prediction endpoint in <code>demo/app.py</code></li>
                <li>Create visualization in <code>demo/static/script.js</code></li>
            </ol>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><strong>GET /</strong> - This page</li>
                <li><strong>GET /health</strong> - Health check</li>
                <li><strong>POST /predict</strong> - Make predictions (coming soon)</li>
            </ul>
        </div>
    </body>
    </html>
    '''

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    port = int(os.getenv('DEMO_PORT', 5000))
    host = os.getenv('DEMO_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    app.run(host=host, port=port, debug=debug)
EOF
    echo "   ✅ Minimal demo created"
    echo ""
fi

# Launch Flask
python demo/app.py

# This only executes if server is stopped
echo ""
echo "🛑 Demo server stopped"
echo ""
