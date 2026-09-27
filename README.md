# mlx-voice-clone

Local **voice-cloned TTS** on macOS (Apple Silicon) via [mlx-audio](https://github.com/Blaizzy/mlx-audio) + Qwen3.

Clone from a short reference clip, then narrate plain-text scripts — CLI or browser UI.

## Requirements

- macOS + Apple Silicon
- [uv](https://github.com/astral-sh/uv) (or Python 3.12+)
- `ffmpeg` on `PATH` (`brew install ffmpeg`)
- Network on first run (Hugging Face model download)

## Quick start

```bash
cd mlx-voice-clone
uv sync

# Record a fresh reference clip (~15s) — speak the lines in documents/ref_voice_transcript.txt
./scripts/record_sample.sh

# CLI
uv run python scripts/qwen3_clone.py \
  --ref-audio ./sample_clip.m4a \
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
mlx-voice-clone/
├── sample_clip.m4a                 # Reference voice (short)
├── documents/
│   ├── ref_voice_transcript.txt    # Exact words in sample_clip (skips Whisper)
│   └── 01_intro.txt                # Demo narration script
├── scripts/
│   ├── qwen3_clone.py              # CLI: synthesis, chunking, ffmpeg export
│   ├── tts_ui_server.py            # FastAPI + SSE generate stream
│   ├── record_sample.sh            # Record sample_clip from the mic
│   └── bench_clone.py              # Optional batch benchmark
├── web/tts/index.html              # Browser UI (no build step)
└── outputs/qwen3_clone/            # Generated audio (+ `_cache/`)  [gitignored]
```

| Piece | Role |
|--------|------|
| `scripts/qwen3_clone.py` | Inference source of truth |
| `scripts/tts_ui_server.py` | Thin wrapper: runs the CLI as a subprocess, streams logs |
| `web/tts/index.html` | Frontend; `/api/*` on the same origin |

**Output naming:** `HHMMSS_<script_stem>.m4a`

## Reference voice

1. Edit `documents/ref_voice_transcript.txt` to the lines you will speak (~10–20s works well).
2. Run `./scripts/record_sample.sh` (dialog → speak → writes `sample_clip.m4a`).
3. Keep the transcript matched to the clip so synthesis can skip Whisper.

## CLI notes

- **`--ref-text-file`** skips Whisper on the reference clip (faster, repeatable).
- Ref audio is converted to **24 kHz mono WAV** and cached under `outputs/qwen3_clone/_cache/`.
- **Stdout:** last line is the output path. **Stderr:** stats / duration.
- **Quality:** `lite` (0.6B) for iteration, `pro` (1.7B) when you care about quality.
- **Export:** default AAC `.m4a` via ffmpeg. Raw WAV: `--output-format wav`.

## Web UI

```bash
uv run python scripts/tts_ui_server.py
```

Open **http://127.0.0.1:8765/** — pick a script, generate, optional goal length / post-process.

Speed levers: `lite` vs `pro`, chunk size, `--fast`, ref text file, and cache. The UI does not make the model faster.

## Adding a narration script

Drop a `.txt` in `documents/`, then pass `--text-file documents/your_script.txt` or refresh the UI list.

## Benchmark

```bash
uv run python scripts/bench_clone.py
uv run python scripts/bench_clone.py --only 01_intro
```

## Troubleshooting

**“Network error” / stream dies but a file still appears**  
Generate uses a long-lived SSE stream. If the Mac sleeps or the browser suspends the tab, the connection can drop after the subprocess already wrote the file. Refresh the page, hit **Refresh** on the output list, or use the CLI. For long runs, keep the machine awake.

**Pitch** (separate from speaking speed): edit the exported file externally without re-synthesizing.
