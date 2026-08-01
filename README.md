# Voice Clone

Local **voice-cloned TTS** on macOS (Apple Silicon) using [mlx-audio](https://github.com/Blaizzy/mlx-audio) and Qwen3. Speak from a short reference clip, then generate narration from plain text — CLI or browser UI.

## Requirements

- macOS + Apple Silicon
- [uv](https://github.com/astral-sh/uv) (or Python 3.12+)
- `ffmpeg` on `PATH` (Homebrew: `brew install ffmpeg`)
- Network on first run (Hugging Face model download)

## Quick start

```bash
cd voice-clone
uv sync

# CLI
uv run python scripts/qwen3_clone.py \
  --ref-audio ./securities_clip.m4a \
  --ref-text-file documents/ref_voice_transcript.txt \
  --text-file documents/01_intro.txt \
  --quality lite \
  --chunk-chars 450 \
  --temperature 0.45 \
  --speed 1.08

# Web UI → http://127.0.0.1:8765/
uv run python scripts/tts_ui_server.py
```

## Layout

```
voice-clone/
├── web/tts/index.html           # Browser UI (no build step)
├── scripts/
│   ├── qwen3_clone.py           # CLI: synthesis, chunking, ffmpeg export
│   ├── tts_ui_server.py         # FastAPI + SSE generate stream
│   └── bench_clone.py           # Optional batch benchmark
├── documents/                   # Narration scripts + ref transcript
│   ├── ref_voice_transcript.txt # Words spoken in the ref clip
│   └── *.txt                    # Scripts (UI lists newest first)
├── securities_clip.m4a          # Default reference voice
└── outputs/qwen3_clone/         # Generated audio (+ `_cache/`)
```

| Piece | Role |
|--------|------|
| `scripts/qwen3_clone.py` | Inference source of truth — models, speed, chunks, export |
| `scripts/tts_ui_server.py` | Thin wrapper: runs the CLI as a subprocess, streams logs |
| `web/tts/index.html` | Frontend; calls `/api/*` on the same origin |

**Output naming:** `HHMMSS_<script_stem>.m4a` (time prefix + script stem).

## CLI details

- **`--ref-text-file`** skips Whisper on the reference clip (faster, repeatable).
- Reference audio is converted to **24 kHz mono WAV** and cached under `outputs/qwen3_clone/_cache/`.
- **Stdout:** last line is the output path. **Stderr:** stats / duration / optional target hint.
- **Target length (hint only):** `--target-seconds 180` prints a suggested next `--speed`; re-run and listen.
- **Export:** default AAC `.m4a` via ffmpeg. Raw WAV: `--output-format wav`. Keep both: `--output-format m4a --keep-wav`.
- **Quality:** `lite` (0.6B) vs `pro` (1.7B). Use `lite` for iteration.

## Web UI

```bash
uv run python scripts/tts_ui_server.py
```

Open **http://127.0.0.1:8765/** — pick a script, preview, generate, optional goal length, post-process (suggest speed / ffmpeg stretch).

The UI does **not** make the model faster. Speed levers remain: `lite` vs `pro`, chunk size, `--fast`, ref text file, and cache.

## Benchmark

```bash
uv run python scripts/bench_clone.py
uv run python scripts/bench_clone.py --only 01_intro
```

## Where to edit

| Goal | File |
|------|------|
| UI / UX | `web/tts/index.html` |
| API / subprocess wiring | `scripts/tts_ui_server.py` |
| Model / audio pipeline | `scripts/qwen3_clone.py` |

Use the web UI to smoke-test; use the CLI for automation and long batches.

## Troubleshooting

**“Network error” / stream dies but a file still appears**  
Generate uses a long-lived SSE stream. If the Mac sleeps or the browser suspends the tab, the connection can drop after the subprocess already wrote the file. Refresh the page, hit **Refresh** on the output list, or use the CLI. For long runs, keep the machine awake (`caffeinate` or Energy settings).

**Pitch** (separate from speaking speed): edit the exported file externally (e.g. a pitch shifter) without re-synthesizing.
