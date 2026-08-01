# Voice clone (Qwen3 TTS, MLX)

Local **voice-clone TTS** on macOS using `mlx-audio` and a short reference clip. This doc is the map for **where things live** and **how to run** them.

## Layout

```
voice-clone/
├── web/tts/index.html           # Browser UI (HTML + CSS + JS, no build step)
├── scripts/
│   ├── qwen3_clone.py           # CLI: synthesis, chunking, ffmpeg export
│   ├── tts_ui_server.py         # FastAPI: serves UI + /api/* (SSE generate)
│   └── bench_clone.py           # Optional: batch benchmark over documents/*.txt
├── documents/                   # Script `.txt` files + ref transcript
│   ├── ref_voice_transcript.txt # Words in ref audio (skips Whisper if set)
│   └── *.txt                    # Your narration scripts (UI lists newest first)
├── securities_clip.m4a          # Default reference voice (project root)
└── outputs/qwen3_clone/         # Generated `.m4a`, ffmpeg cache under `_cache/`
```

| Piece | Role |
|--------|------|
| **`scripts/qwen3_clone.py`** | Source of truth for inference: models, speed, chunks, targets. |
| **`scripts/tts_ui_server.py`** | Thin wrapper: runs the CLI as a subprocess, streams logs via SSE. |
| **`web/tts/index.html`** | All frontend behavior; calls `/api/*` on the same origin. |

**Naming outputs:** files look like `HHMMSS_<script_stem>.m4a` (time prefix + script filename stem). Rename in Finder if you want friendlier names.

## Setup

```bash
cd /path/to/voice-clone
uv sync   # or: uv pip install -e .
```

## CLI (fastest path)

```bash
cd /path/to/voice-clone

uv run python scripts/qwen3_clone.py \
  --ref-audio ./securities_clip.m4a \
  --ref-text-file documents/ref_voice_transcript.txt \
  --text-file documents/01_intro.txt \
  --quality lite \
  --chunk-chars 450 \
  --temperature 0.45 \
  --speed 1.08
```

- **`--ref-text-file`** avoids Whisper on the reference clip (faster, repeatable).
- Reference audio is converted to **24 kHz mono WAV** and **cached** under `outputs/qwen3_clone/_cache/`.
- **Stdout:** last line is the output file path (good for scripts). **Stderr:** stats / duration / optional target hint.

**Target length (hint only):** `--target-seconds 180` prints a suggested next `--speed`; re-run and listen.

**Export:** default **AAC `.m4a`** via ffmpeg. Raw WAV: `--output-format wav`. Keep WAV next to m4a: `--output-format m4a --keep-wav`.

## Web UI

```bash
uv run python scripts/tts_ui_server.py
```

Open **http://127.0.0.1:8765/** — script picker + preview, generate, optional goal length, post-process (suggest speed / ffmpeg stretch).

The UI does **not** make the model faster; it’s convenience. Speed levers are still: `lite` vs `pro`, chunk size, `--fast`, ref text file, cache.

## Benchmark (optional)

```bash
uv run python scripts/bench_clone.py
uv run python scripts/bench_clone.py --only 01_intro
```

## Improving the tool

- **UI / UX:** edit **`web/tts/index.html`** only (unless you add new API routes).
- **API or subprocess wiring:** edit **`scripts/tts_ui_server.py`**.
- **Model / audio pipeline:** edit **`scripts/qwen3_clone.py`**.
- Keep the web UI around if you want to **smoke-test** features quickly; use **CLI** for automation and long batches.

## Troubleshooting

**“Network error” / stream dies but a file still appears on disk**  
Generate uses a **long-lived HTTP stream** (SSE). If the **Mac sleeps** or the **browser suspends**, the connection can drop after the subprocess has already written the file. Refresh the page, hit **Refresh** on the output list, or use the CLI. For long runs, keep the machine awake (`caffeinate` or Energy settings).

**Pitch** (separate from speaking speed): e.g. [vocalremover.org/pitch](https://vocalremover.org/pitch) on the exported file without re-synthesizing.
