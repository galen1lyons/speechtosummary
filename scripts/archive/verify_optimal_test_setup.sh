#!/bin/bash
# Verification script for Optimal Whisper Test setup
# Run this before executing the autonomous test suite

set -e

echo "=========================================="
echo "Optimal Whisper Test - Setup Verification"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Working directory
echo -n "1. Checking working directory... "
if [ -f "scripts/optimal_whisper_test.py" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Please run from: /home/dedmtiintern/speechtosummary"
    exit 1
fi

# Check 2: Virtual environment
echo -n "2. Checking virtual environment... "
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo -e "${GREEN}✓${NC}"
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        echo -e "   ${YELLOW}⚠${NC}  Virtual environment not activated"
        echo "   Run: source venv/bin/activate"
    else
        echo "   Virtual environment: $VIRTUAL_ENV"
    fi
else
    echo -e "${RED}✗${NC}"
    echo "   Virtual environment not found at: venv/"
    exit 1
fi

# Check 3: Audio file
echo -n "3. Checking audio file... "
AUDIO_FILE="data/mamak session scam.mp3"
if [ -f "$AUDIO_FILE" ]; then
    SIZE=$(du -h "$AUDIO_FILE" | cut -f1)
    echo -e "${GREEN}✓${NC}"
    echo "   File: $AUDIO_FILE ($SIZE)"
else
    echo -e "${RED}✗${NC}"
    echo "   Audio file not found: $AUDIO_FILE"
    echo "   Available audio files:"
    ls -lh data/*.mp3 2>/dev/null || echo "   No audio files found in data/"
    exit 1
fi

# Check 4: Python dependencies
echo -n "4. Checking Python dependencies... "
if python -c "import whisper, transformers, torch" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Missing dependencies. Install with:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check 5: Script syntax
echo -n "5. Checking script syntax... "
if python3 -m py_compile scripts/optimal_whisper_test.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Syntax error in optimal_whisper_test.py"
    exit 1
fi

# Check 6: Disk space
echo -n "6. Checking disk space... "
AVAILABLE=$(df -BM . | tail -1 | awk '{print $4}' | sed 's/M//')
if [ "$AVAILABLE" -gt 500 ]; then
    echo -e "${GREEN}✓${NC}"
    echo "   Available: ${AVAILABLE}M"
else
    echo -e "${YELLOW}⚠${NC}"
    echo "   Low disk space: ${AVAILABLE}M (recommend >500M)"
fi

# Check 7: Output directories
echo -n "7. Checking output directories... "
mkdir -p outputs/optimal_test
echo -e "${GREEN}✓${NC}"
echo "   Created: outputs/optimal_test/"

echo ""
echo "=========================================="
echo -e "${GREEN}Setup verification complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv:     source venv/bin/activate"
echo "  2. Preview tests:     python scripts/optimal_whisper_test.py --audio \"$AUDIO_FILE\" --dry-run"
echo "  3. Run tests:         python scripts/optimal_whisper_test.py --audio \"$AUDIO_FILE\""
echo ""
echo "Estimated runtime: 40-60 minutes for 8 tests"
echo ""
