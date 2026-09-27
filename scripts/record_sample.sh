#!/usr/bin/env bash
# Record sample_clip.m4a from the default Mac mic.
# Speak the lines in documents/ref_voice_transcript.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/sample_clip.m4a"
SECONDS_N="${1:-16}"
SCRIPT_FILE="$ROOT/documents/ref_voice_transcript.txt"
SCRIPT="$(tr '\n' ' ' < "$SCRIPT_FILE" | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')"

osascript <<EOF
display dialog "Click OK, then speak clearly for ${SECONDS_N} seconds:

${SCRIPT}" buttons {"Cancel", "Record"} default button "Record" with title "Record sample_clip"
EOF

echo "Recording ${SECONDS_N}s → $OUT"
# :0 = MacBook Pro Microphone (see: ffmpeg -f avfoundation -list_devices true -i "")
ffmpeg -y -hide_banner -loglevel error \
  -f avfoundation -i ":0" \
  -t "$SECONDS_N" \
  -ac 1 -ar 24000 \
  -c:a aac -b:a 128k \
  "$OUT"

echo "Wrote $OUT ($(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT")s)"
echo "Confirm the transcript still matches: documents/ref_voice_transcript.txt"
