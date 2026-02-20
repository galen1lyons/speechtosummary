#!/bin/bash
# Ground Truth Transcription Helper
# Makes the transcription process easier with audio playback controls

AUDIO_FILE="data/mamak session scam.mp3"
GROUND_TRUTH_FILE="data/ground_truth_mamak.txt"

echo "=================================================="
echo "Ground Truth Transcription Helper"
echo "=================================================="
echo ""
echo "Audio file: $AUDIO_FILE"
echo "Ground truth output: $GROUND_TRUTH_FILE"
echo ""

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "❌ Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Get audio duration
duration=$(ffprobe -i "$AUDIO_FILE" -show_entries format=duration -v quiet -of csv="p=0" 2>/dev/null)
minutes=$(echo "$duration / 60" | bc)
echo "Audio duration: ${minutes} minutes"
echo ""

echo "=================================================="
echo "Recommended Approach:"
echo "=================================================="
echo ""
echo "OPTION 1: Sample Sections (30-60 min) - RECOMMENDED"
echo "  Transcribe 3 representative sections:"
echo "    • Section 1: 0:00 - 2:00  (intro, clear audio)"
echo "    • Section 2: 6:00 - 8:00  (phone call, challenging)"
echo "    • Section 3: 13:00 - 15:00 (outro, clear audio)"
echo ""
echo "OPTION 2: Full Transcription (3-5 hours)"
echo "  Transcribe entire 15-minute audio"
echo ""
echo "=================================================="
echo "Choose your approach:"
echo "=================================================="
echo "1) Sample sections (recommended)"
echo "2) Full transcription"
echo "3) Custom time range"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "✅ Sample sections approach selected"
        echo ""
        echo "Creating template file: $GROUND_TRUTH_FILE"
        cat > "$GROUND_TRUTH_FILE" << 'EOF'
# Ground Truth Transcript - Sample Sections
# Audio: mamak session scam.mp3
# Transcriber: [Your Name]
# Date: [Date]

# ===== SECTION 1: Intro (0:00 - 2:00) =====

[Transcribe what you hear here]


# ===== SECTION 2: Phone Call (6:00 - 8:00) =====

[Transcribe what you hear here]


# ===== SECTION 3: Outro (13:00 - 15:00) =====

[Transcribe what you hear here]


# END OF TRANSCRIPT
# Notes:
# - Mark unclear sections with [unclear: word?]
# - Mark unintelligible with [unintelligible]
# - Include filler words (um, uh, like)
EOF
        echo "✅ Template created!"
        echo ""
        echo "Next steps:"
        echo "1. Open editor: nano $GROUND_TRUTH_FILE"
        echo "2. Play Section 1: ffplay -ss 0 -t 120 \"$AUDIO_FILE\""
        echo "3. Play Section 2: ffplay -ss 360 -t 120 \"$AUDIO_FILE\""
        echo "4. Play Section 3: ffplay -ss 780 -t 120 \"$AUDIO_FILE\""
        echo ""
        echo "Tip: Use Space to pause/play, Q to quit player"
        ;;

    2)
        echo ""
        echo "✅ Full transcription approach selected"
        echo ""
        echo "Creating template file: $GROUND_TRUTH_FILE"
        cat > "$GROUND_TRUTH_FILE" << 'EOF'
# Ground Truth Transcript - Full Audio
# Audio: mamak session scam.mp3
# Duration: 15:22
# Transcriber: [Your Name]
# Date: [Date]

[Start transcribing from 0:00]


# END OF TRANSCRIPT
# Notes:
# - Mark unclear sections with [unclear: word?]
# - Mark unintelligible with [unintelligible]
# - Include filler words (um, uh, like)
# - Take breaks every 15-20 minutes!
EOF
        echo "✅ Template created!"
        echo ""
        echo "Next steps:"
        echo "1. Open editor: nano $GROUND_TRUTH_FILE"
        echo "2. Play audio: ffplay \"$AUDIO_FILE\""
        echo ""
        echo "Tip: Use Space to pause/play, Left/Right arrows to rewind/forward"
        ;;

    3)
        echo ""
        read -p "Start time (MM:SS format, e.g., 02:30): " start_time
        read -p "Duration in seconds (e.g., 120 for 2 minutes): " duration_sec

        echo ""
        echo "✅ Custom range selected: $start_time for ${duration_sec}s"
        echo ""
        echo "Creating template file: $GROUND_TRUTH_FILE"
        cat > "$GROUND_TRUTH_FILE" << EOF
# Ground Truth Transcript - Custom Section
# Audio: mamak session scam.mp3
# Section: $start_time for ${duration_sec}s
# Transcriber: [Your Name]
# Date: [Date]

[Transcribe what you hear here]


# END OF TRANSCRIPT
EOF
        echo "✅ Template created!"
        echo ""

        # Convert MM:SS to seconds
        start_seconds=$(echo "$start_time" | awk -F: '{print ($1 * 60) + $2}')

        echo "Next steps:"
        echo "1. Open editor: nano $GROUND_TRUTH_FILE"
        echo "2. Play section: ffplay -ss $start_seconds -t $duration_sec \"$AUDIO_FILE\""
        ;;

    4)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Transcription Tips:"
echo "=================================================="
echo ""
echo "✅ DO:"
echo "  • Listen carefully and transcribe exactly what you hear"
echo "  • Include filler words (um, uh, like)"
echo "  • Include natural repetitions"
echo "  • Use simple punctuation (. , ?)"
echo "  • Mark unclear parts as [unclear: word?]"
echo ""
echo "❌ DON'T:"
echo "  • Don't clean up or fix the speech"
echo "  • Don't skip unclear sections"
echo "  • Don't copy from model output"
echo "  • Don't add words that weren't said"
echo ""
echo "📝 Keyboard Shortcuts (ffplay):"
echo "  • Space: Pause/Play"
echo "  • Left/Right: Rewind/Forward 10s"
echo "  • Up/Down: Rewind/Forward 60s"
echo "  • Q: Quit player"
echo ""
echo "=================================================="
echo "Ready to start transcribing!"
echo "=================================================="
