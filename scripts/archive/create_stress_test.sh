#!/bin/bash
# create_stress_test.sh - Create a challenging multi-source audio mix for Malaysian Whisper testing

cd /home/dedmtiintern/speechtosummary/data

ffmpeg -i base_mamak.mp3 \
       -i interference_chinese.mp3 \
       -i chaos_parliament.mp3 \
       -filter_complex \
       "[0:a]atrim=0:600,asetpts=PTS-STARTPTS,volume=1.0[main]; \
        [1:a]atrim=60:540,asetpts=PTS-STARTPTS,highpass=f=300,lowpass=f=3400,volume=0.6,pan=stereo|c0=0.8*c0|c1=0.2*c1,adelay=120000|120000[phone]; \
        [2:a]atrim=30:270,asetpts=PTS-STARTPTS,aecho=0.8:0.9:40:0.5,volume=0.5,pan=stereo|c0=0.2*c0|c1=0.8*c1,adelay=360000|360000[reverb]; \
        [main][phone][reverb]amix=inputs=3:duration=first:dropout_transition=0,dynaudnorm[out]" \
       -map "[out]" \
       malaysian_whisper_stress_test.mp3

echo "Stress test audio created: malaysian_whisper_stress_test.mp3"
