#!/bin/bash

# ============================================
# CortexCore - ENVIRONMENT SETUP
# ============================================
# Automated setup script for rapid hackathon development
# Usage: bash scripts/01_setup_environment.sh
# Time: ~5-10 minutes depending on internet speed

set -e  # Exit on any error

echo "🚀 ============================================"
echo "🚀 CortexCore - SETUP"
echo "🚀 ============================================"
echo ""

# ==========================================
# 1. Check Python Version
# ==========================================

echo "📋 Step 1/8: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

echo "   Found: Python $python_version"

if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 10 ]; then
    echo "   ✅ Python version is compatible (3.10+)"
else
    echo "   ❌ ERROR: Python 3.10 or 3.11 required"
    echo "   Current: Python $python_version"
    echo "   Please install Python 3.10+ and try again"
    exit 1
fi

echo ""

# ==========================================
# 2. Create Virtual Environment
# ==========================================

echo "📦 Step 2/8: Creating virtual environment..."

if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists"
    read -p "   Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing old virtual environment..."
        rm -rf venv
        echo "   Creating new virtual environment..."
        python3 -m venv venv
    else
        echo "   Using existing virtual environment"
    fi
else
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

echo "   ✅ Virtual environment ready"
echo ""

# ==========================================
# 3. Activate Virtual Environment
# ==========================================

echo "🔌 Step 3/8: Activating virtual environment..."

# Detect OS for activation
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
    echo "   ✅ Activated (Windows)"
else
    # Unix-like (Linux, macOS)
    source venv/bin/activate
    echo "   ✅ Activated (Unix/Mac)"
fi

echo ""

# ==========================================
# 4. Upgrade pip
# ==========================================

echo "⬆️  Step 4/8: Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "   ✅ pip upgraded to $(pip --version | awk '{print $2}')"
echo ""

# ==========================================
# 5. Install PyTorch (with CUDA if available)
# ==========================================

echo "🔥 Step 5/8: Installing PyTorch..."
echo "   Checking for CUDA availability..."

# Check if nvidia-smi exists (indicates NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ NVIDIA GPU detected, installing CUDA version"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
    echo "   ✅ PyTorch installed with CUDA 11.8 support"
else
    echo "   ℹ️  No NVIDIA GPU detected, installing CPU version"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    echo "   ✅ PyTorch installed (CPU only)"
fi

echo ""

# ==========================================
# 6. Install Core Dependencies
# ==========================================

echo "📚 Step 6/8: Installing core dependencies..."
echo "   This may take 2-5 minutes..."

# Install from requirements.txt
pip install -r requirements.txt --quiet

echo "   ✅ All dependencies installed"
echo ""

# ==========================================
# 7. Verify Installation
# ==========================================

echo "🔍 Step 7/8: Verifying installation..."

# Create verification script
cat > /tmp/verify_install.py << 'EOF'
import sys

try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      CUDA version: {torch.version.cuda}")
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    import snntorch
    print(f"   ✅ snnTorch {snntorch.__version__}")

    import neurokit2
    print(f"   ✅ neurokit2 installed")

    import flask
    print(f"   ✅ Flask {flask.__version__}")

    import plotly
    print(f"   ✅ Plotly {plotly.__version__}")

    import jupyter
    print(f"   ✅ Jupyter installed")

    print("\n   🎉 All core packages installed successfully!")

except ImportError as e:
    print(f"   ❌ ERROR: {e}")
    print("   Please check requirements.txt and try again")
    sys.exit(1)
EOF

python /tmp/verify_install.py
rm /tmp/verify_install.py

echo ""

# ==========================================
# 8. Create Directory Structure
# ==========================================

echo "📁 Step 8/8: Setting up directory structure..."

# Create directories if they don't exist
mkdir -p data/synthetic
mkdir -p data/cache
mkdir -p models
mkdir -p results/plots
mkdir -p results/metrics

# Create .gitkeep files to track empty directories
touch data/synthetic/.gitkeep
touch data/cache/.gitkeep
touch models/.gitkeep
touch results/plots/.gitkeep
touch results/metrics/.gitkeep

# Create .env from .env.example if it exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "   Creating .env from .env.example..."
        cp .env.example .env
        echo "   ⚠️  Remember to configure .env with your settings"
    else
        echo "   ⚠️  .env.example not found, skipping .env creation"
    fi
fi

echo "   ✅ Directory structure ready"
echo ""

# ==========================================
# Final Instructions
# ==========================================

echo "🎉 ============================================"
echo "🎉 SETUP COMPLETE!"
echo "🎉 ============================================"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Configure your environment (optional):"
echo "   nano .env  # or use your favorite editor"
echo ""
echo "3. Generate MVP dataset:"
echo "   bash scripts/02_generate_mvp_data.sh"
echo ""
echo "4. Start Jupyter for exploration:"
echo "   jupyter notebook"
echo ""
echo "5. Or jump straight to training:"
echo "   bash scripts/03_train_mvp_model.sh"
echo ""
echo "📚 Documentation: See README.md"
echo "🐛 Troubleshooting: Check logs/ directory"
echo "💬 Questions: Ask your team lead"
echo ""
echo "✨ Happy hacking! Let's build something amazing! ✨"
echo ""
