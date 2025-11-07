#!/bin/bash

echo "========================================"
echo "CortexCore Demo Verification Script"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if demo is running
if ! curl -s http://localhost:5000/health > /dev/null; then
    echo -e "${RED}✗ Demo is not running${NC}"
    echo "  Run: make demo"
    exit 1
fi

echo -e "${GREEN}✓ Demo is running${NC}"

# Test 1: Model load status
echo ""
echo "Test 1: Model Load Status"
MODEL_LOADED=$(curl -s http://localhost:5000/health | grep -o '"loaded":[^,}]*' | cut -d':' -f2)
if [ "$MODEL_LOADED" = "true" ]; then
    echo -e "${GREEN}✓ Model loaded successfully${NC}"
else
    echo -e "${RED}✗ Model failed to load${NC}"
    curl -s http://localhost:5000/health | python -m json.tool
    exit 1
fi

# Test 2: Prediction is not random
echo ""
echo "Test 2: Prediction Accuracy"
RESPONSE=$(curl -s -X POST http://localhost:5000/api/generate_sample \
  -H "Content-Type: application/json" \
  -d '{"condition": "normal"}')

PREDICTION=$(echo "$RESPONSE" | curl -s -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @- | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('confidence', 0))")

if [ "$PREDICTION" = "0.5" ]; then
    echo -e "${RED}✗ Prediction is 50% (random guess)${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Prediction is not random (confidence: $PREDICTION)${NC}"
fi

# Test 3: Spike count is reasonable
echo ""
echo "Test 3: Spike Encoding"
SPIKE_COUNT=$(echo "$RESPONSE" | curl -s -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @- | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('spike_count', 0))")

if [ "$SPIKE_COUNT" -gt "100" ] && [ "$SPIKE_COUNT" -lt "25000" ]; then
    echo -e "${GREEN}✓ Spike count is reasonable ($SPIKE_COUNT)${NC}"
else
    echo -e "${YELLOW}⚠ Spike count might be incorrect ($SPIKE_COUNT)${NC}"
fi

# Test 4: Different predictions for different conditions
echo ""
echo "Test 4: Model Discrimination"
NORMAL_CONF=$(curl -s -X POST http://localhost:5000/api/generate_sample \
  -H "Content-Type: application/json" \
  -d '{"condition": "normal"}' | \
  curl -s -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @- | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('confidence', 0))")

ARR_CONF=$(curl -s -X POST http://localhost:5000/api/generate_sample \
  -H "Content-Type: application/json" \
  -d '{"condition": "arrhythmia"}' | \
  curl -s -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @- | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('confidence', 0))")

echo "  Normal confidence: $NORMAL_CONF"
echo "  Arrhythmia confidence: $ARR_CONF"

if [ "$NORMAL_CONF" != "$ARR_CONF" ]; then
    echo -e "${GREEN}✓ Model produces different predictions${NC}"
else
    echo -e "${YELLOW}⚠ Predictions might be identical${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}All critical tests passed!${NC}"
echo "========================================"
