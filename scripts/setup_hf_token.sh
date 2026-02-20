#!/bin/bash

# Hugging Face Token Setup Script
# This script helps you set up your HF token for speaker diarization

echo "=========================================="
echo "Hugging Face Token Setup"
echo "=========================================="
echo ""

# Check if token is already set
if [ -n "$HF_TOKEN" ]; then
    echo "✅ HF_TOKEN is already set!"
    echo "Current token: ${HF_TOKEN:0:10}..."
    echo ""
else
    echo "⚠️  HF_TOKEN is not set yet."
    echo ""
fi

echo "To set your token, run ONE of the following:"
echo ""
echo "Option 1: Temporary (current session only)"
echo "  export HF_TOKEN=your_token_here"
echo ""
echo "Option 2: Permanent (recommended)"
echo "  echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "Option 3: Use .env file (most secure)"
echo "  echo 'HF_TOKEN=your_token_here' >> .env"
echo "  (Code will automatically load from .env)"
echo ""
echo "=========================================="
echo ""

# Test if token works
if [ -n "$HF_TOKEN" ]; then
    echo "Testing token..."
    python3 -c "
import os
token = os.getenv('HF_TOKEN')
if token:
    print(f'✅ Token detected: {token[:10]}...')
    print('✅ Ready to use!')
else:
    print('❌ Token not accessible from Python')
" 2>/dev/null || echo "⚠️  Python test failed (this is OK if you haven't set it yet)"
fi
