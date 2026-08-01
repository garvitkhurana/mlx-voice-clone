# Docs

Start with the root [README](../README.md) — setup, CLI, web UI, and troubleshooting.

## Adding a narration script

1. Drop a `.txt` file in `documents/`
2. Run the CLI with `--text-file documents/your_script.txt`, or refresh the web UI script list

## Reference voice

- Default clip: `securities_clip.m4a` at the repo root
- Put the spoken words in `documents/ref_voice_transcript.txt` so synthesis can skip Whisper on the reference
