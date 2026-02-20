#!/bin/bash
# Convenient wrapper script for running comprehensive tests
# This script automatically activates the virtual environment

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Comprehensive Whisper Testing - Launcher"
echo "========================================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✅ Activating virtual environment...${NC}"
source venv/bin/activate

# Validate setup
echo ""
echo -e "${YELLOW}Running setup validation...${NC}"
python scripts/validate_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Setup validation failed. Please fix the issues above.${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  Ready to run tests!"
echo "========================================================================"
echo ""
echo "Choose an option:"
echo "  1) Quick validation test (2 tests, ~10-20 minutes)"
echo "  2) Small test suite (5 tests, ~30-60 minutes)"
echo "  3) Full test suite (36 tests, ~3-6 hours)"
echo "  4) Custom (specify --limit and --start-from)"
echo "  5) Just validate setup (already done above)"
echo "  6) Analyze existing results"
echo "  7) Generate presentation materials"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running quick validation (2 tests)...${NC}\n"
        python scripts/comprehensive_whisper_test.py \
            --audio "data/mamak session scam.mp3" \
            --limit 2
        ;;
    2)
        echo -e "\n${GREEN}Running small test suite (5 tests)...${NC}\n"
        python scripts/comprehensive_whisper_test.py \
            --audio "data/mamak session scam.mp3" \
            --limit 5
        ;;
    3)
        echo -e "\n${GREEN}Running full test suite (36 tests)...${NC}\n"
        echo -e "${YELLOW}⚠️  This will take 3-6 hours. Consider running overnight.${NC}\n"
        read -p "Press ENTER to continue or Ctrl+C to cancel..."
        python scripts/comprehensive_whisper_test.py \
            --audio "data/mamak session scam.mp3"
        ;;
    4)
        echo ""
        read -p "Enter --limit (number of tests, or leave empty for all): " limit
        read -p "Enter --start-from (test ID like BS1, or leave empty): " start_from

        cmd="python scripts/comprehensive_whisper_test.py --audio \"data/mamak session scam.mp3\""

        if [ -n "$limit" ]; then
            cmd="$cmd --limit $limit"
        fi

        if [ -n "$start_from" ]; then
            cmd="$cmd --start-from $start_from"
        fi

        echo -e "\n${GREEN}Running: $cmd${NC}\n"
        eval $cmd
        ;;
    5)
        echo -e "\n${GREEN}✅ Setup validation complete!${NC}"
        exit 0
        ;;
    6)
        echo -e "\n${GREEN}Analyzing results...${NC}\n"
        python scripts/analyze_test_results.py
        ;;
    7)
        echo -e "\n${GREEN}Generating presentation materials...${NC}\n"
        python scripts/generate_presentation.py
        ;;
    *)
        echo -e "\n${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# If tests were run, ask about analysis
if [ $choice -le 4 ]; then
    echo ""
    echo "========================================================================"
    read -p "Run analysis now? [y/N]: " run_analysis

    if [[ $run_analysis =~ ^[Yy]$ ]]; then
        echo -e "\n${GREEN}Analyzing results...${NC}\n"
        python scripts/analyze_test_results.py

        echo ""
        read -p "Generate presentation materials? [y/N]: " gen_pres

        if [[ $gen_pres =~ ^[Yy]$ ]]; then
            echo -e "\n${GREEN}Generating presentation...${NC}\n"
            python scripts/generate_presentation.py

            echo ""
            echo -e "${GREEN}✅ All done!${NC}"
            echo "Check results/comprehensive_test/presentation/ for your materials."
        fi
    fi
fi

echo ""
echo "========================================================================"
echo "  Complete!"
echo "========================================================================"
