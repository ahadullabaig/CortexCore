#!/bin/bash

# Dry-run test for make quick-start
# Checks all prerequisites without running full pipeline

set -e

echo "🧪 Testing make quick-start prerequisites..."
echo ""

# Check 1: In project root
if [ ! -f "Makefile" ]; then
    echo "❌ ERROR: Not in project root (no Makefile found)"
    echo "   Run from: $(dirname $(realpath $0))"
    exit 1
fi
echo "✅ Running from project root"

# Check 2: Python version
python3 --version 2>&1 | grep -E "Python 3\.(1[0-9]|[2-9][0-9])" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Python 3.10+ available"
else
    echo "❌ ERROR: Python 3.10+ required"
    exit 1
fi

# Check 3: Disk space (need ~2GB)
available=$(df . | tail -1 | awk '{print $4}')
if [ $available -gt 2000000 ]; then
    echo "✅ Sufficient disk space"
else
    echo "⚠️  WARNING: Low disk space (need ~2GB)"
fi

# Check 4: Port 5000 availability
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  WARNING: Port 5000 already in use"
else
    echo "✅ Port 5000 available"
fi

# Check 5: GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
else
    echo "⚠️  INFO: No GPU detected, training will use CPU (slower)"
fi

echo ""
echo "🎉 Prerequisites check passed!"
echo ""
echo "📝 Next steps:"
echo "   make quick-start  # Takes 13-15 min (or 30-60 min on CPU)"
echo ""
