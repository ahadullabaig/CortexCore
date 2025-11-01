#!/bin/bash

# ============================================
# CortexCore - INTEGRATION TESTS
# ============================================
# Tests end-to-end pipeline
# Usage: bash scripts/05_test_integration.sh
# Time: ~30 seconds

set -e  # Exit on error

echo "🧪 ============================================"
echo "🧪 INTEGRATION TESTS"
echo "🧪 ============================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# ==========================================
# Helper Functions
# ==========================================

run_test() {
    local test_name="$1"
    local test_command="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "   Testing: $test_name... "

    if eval "$test_command" &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ==========================================
# Test 1: Environment Setup
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 1: Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "Python 3.10+ available" "python3 --version | grep -E 'Python 3\.(1[0-9]|[2-9][0-9])'"
run_test "Virtual environment exists" "test -d venv"
run_test "requirements.txt exists" "test -f requirements.txt"
run_test ".env.example exists" "test -f .env.example"
run_test "README.md exists" "test -f README.md"

echo ""

# ==========================================
# Test 2: Python Package Imports
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 2: Python Package Imports"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "PyTorch import" "python -c 'import torch'"
run_test "snnTorch import" "python -c 'import snntorch'"
run_test "neurokit2 import" "python -c 'import neurokit2'"
run_test "Flask import" "python -c 'import flask'"
run_test "NumPy import" "python -c 'import numpy'"
run_test "Pandas import" "python -c 'import pandas'"
run_test "Plotly import" "python -c 'import plotly'"

echo ""

# ==========================================
# Test 3: Directory Structure
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 3: Directory Structure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "src/ directory exists" "test -d src"
run_test "notebooks/ directory exists" "test -d notebooks"
run_test "demo/ directory exists" "test -d demo"
run_test "data/ directory exists" "test -d data"
run_test "models/ directory exists" "test -d models"
run_test "results/ directory exists" "test -d results"
run_test "scripts/ directory exists" "test -d scripts"

echo ""

# ==========================================
# Test 4: Data Pipeline (if data exists)
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 4: Data Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "data/synthetic/mvp_dataset.pt" ]; then
    run_test "MVP dataset exists" "test -f data/synthetic/mvp_dataset.pt"
    run_test "Dataset is loadable" "python -c 'import torch; data = torch.load(\"data/synthetic/mvp_dataset.pt\"); print(data.keys())'"
    run_test "Dataset has train split" "python -c 'import torch; data = torch.load(\"data/synthetic/mvp_dataset.pt\"); assert \"train\" in data'"
    run_test "Dataset has val split" "python -c 'import torch; data = torch.load(\"data/synthetic/mvp_dataset.pt\"); assert \"val\" in data'"
    run_test "Dataset has metadata" "python -c 'import torch; data = torch.load(\"data/synthetic/mvp_dataset.pt\"); assert \"metadata\" in data'"
else
    echo -e "   ${YELLOW}⚠️  SKIP: No dataset found (run scripts/02_generate_mvp_data.sh)${NC}"
fi

echo ""

# ==========================================
# Test 5: Model Pipeline (if model exists)
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 5: Model Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "models/best_model.pt" ]; then
    run_test "Model checkpoint exists" "test -f models/best_model.pt"
    run_test "Model is loadable" "python -c 'import torch; model = torch.load(\"models/best_model.pt\", map_location=\"cpu\")'"
    run_test "Model has state_dict" "python -c 'import torch; data = torch.load(\"models/best_model.pt\", map_location=\"cpu\"); assert \"model_state_dict\" in data or \"state_dict\" in data'"
else
    echo -e "   ${YELLOW}⚠️  SKIP: No model found (run scripts/03_train_mvp_model.sh)${NC}"
fi

echo ""

# ==========================================
# Test 6: Demo Application
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 6: Demo Application"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "demo/app.py exists" "test -f demo/app.py"
run_test "Flask app is importable" "python -c 'import sys; sys.path.insert(0, \"demo\"); import app'"

# Try to start server briefly
echo -n "   Testing: Demo server starts... "
if timeout 5 python demo/app.py &> /tmp/demo_test.log &
then
    SERVER_PID=$!
    sleep 3

    # Check if server is responding
    if curl -s http://localhost:5000/health &> /dev/null || curl -s http://localhost:5000/ &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Kill server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
else
    echo -e "${YELLOW}⚠️  SKIP (server didn't start)${NC}"
fi

rm -f /tmp/demo_test.log

echo ""

# ==========================================
# Test 7: Scripts Executable
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 7: Automation Scripts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "setup script exists" "test -f scripts/01_setup_environment.sh"
run_test "data generation script exists" "test -f scripts/02_generate_mvp_data.sh"
run_test "training script exists" "test -f scripts/03_train_mvp_model.sh"
run_test "demo script exists" "test -f scripts/04_run_demo.sh"
run_test "test script exists" "test -f scripts/05_test_integration.sh"

echo ""

# ==========================================
# Test 8: GPU/CUDA (if available)
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUITE 8: GPU/CUDA Support"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test "CUDA available check" "python -c 'import torch; torch.cuda.is_available()' || true"

if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo -e "   ${GREEN}✅ CUDA is available${NC}"
    python -c 'import torch; print(f"   GPU: {torch.cuda.get_device_name(0)}")'
    python -c 'import torch; print(f"   CUDA version: {torch.version.cuda}")'
else
    echo -e "   ${YELLOW}ℹ️  CUDA not available (CPU only)${NC}"
fi

echo ""

# ==========================================
# Summary
# ==========================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   Total tests: $TOTAL_TESTS"
echo -e "   ${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "   ${RED}Failed: $TESTS_FAILED${NC}"
echo ""

# Calculate percentage
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$((100 * TESTS_PASSED / TOTAL_TESTS))
    echo "   Pass rate: $PASS_RATE%"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
        echo ""
        echo "✅ System is ready for development!"
        echo ""
        echo "Next steps:"
        echo "   1. Generate data: bash scripts/02_generate_mvp_data.sh"
        echo "   2. Train model: bash scripts/03_train_mvp_model.sh"
        echo "   3. Launch demo: bash scripts/04_run_demo.sh"
        exit 0
    else
        echo -e "${RED}❌ SOME TESTS FAILED${NC}"
        echo ""
        echo "Please fix the failed tests before proceeding."
        echo ""
        echo "Common fixes:"
        echo "   - Run setup: bash scripts/01_setup_environment.sh"
        echo "   - Install packages: pip install -r requirements.txt"
        echo "   - Check Python version: python3 --version"
        exit 1
    fi
else
    echo "No tests were run"
    exit 1
fi
