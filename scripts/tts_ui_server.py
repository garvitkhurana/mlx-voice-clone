#!/usr/bin/env python3
"""Local web UI for scripts/qwen3_clone.py (FastAPI + SSE log stream)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web" / "tts"
QWEN_SCRIPT = REPO_ROOT / "scripts" / "qwen3_clone.py"
DEFAULT_REF_TEXT = REPO_ROOT / "documents" / "ref_voice_transcript.txt"
DEFAULT_DOCS = REPO_ROOT / "documents"
DEFAULT_OUT = REPO_ROOT / "outputs" / "qwen3_clone"


def _default_ref_audio() -> Path | None:
    p = REPO_ROOT / "securities_clip.m4a"
    return p if p.is_file() else None


def _ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip())


def _atempo_chain_filter(desired_tempo: float) -> str:
    """desired_tempo = input_duration / target_duration; each atempo in [0.5, 2]."""
    if desired_tempo <= 0:
        raise ValueError("invalid tempo")
    factors: list[float] = []
    x = desired_tempo
    while x > 2.0:
        factors.append(2.0)
        x /= 2.0
    while x < 0.5:
        factors.append(0.5)
        x /= 0.5
    factors.append(x)
    return ",".join(f"atempo={f:.6g}" for f in factors)


def _ffmpeg_fit_duration(src: Path, target_s: float, dst: Path) -> None:
    dur = _ffprobe_duration(src)
    if abs(dur - target_s) < 0.25:
        shutil.copy2(src, dst)
        return
    tempo = dur / target_s
    filt = _atempo_chain_filter(tempo)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-filter:a",
        filt,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


class GenerateBody(BaseModel):
    text_file: str = Field(..., description="Filename under documents/, e.g. 01_intro.txt")
    speed: float = 1.08
    quality: str = "lite"
    chunk_chars: int = 700
    fast: bool = False
    temperature: float = 0.45
    max_tokens: int = 4096
    target_seconds: float | None = None
    fit_ffmpeg: bool = Field(
        False,
        description="If True (and target set), time-stretch output with ffmpeg to match target.",
    )


app = FastAPI(title="Qwen3 clone UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    p = WEB_DIR / "index.html"
    if not p.is_file():
        raise HTTPException(404, "web/tts/index.html missing")
    return FileResponse(p)


@app.get("/api/documents")
def list_documents() -> dict:
    """Basenames of script .txt files under documents/ (newest first), excluding ref-voice transcripts."""
    docs = DEFAULT_DOCS
    if not docs.is_dir():
        return {"documents_dir": "documents", "files": []}
    items: list[tuple[str, float]] = []
    for p in docs.glob("*.txt"):
        if not p.is_file():
            continue
        if "ref_voice" in p.name.lower():
            continue
        try:
            st = p.stat()
            items.append((p.name, st.st_mtime))
        except OSError:
            continue
    items.sort(key=lambda x: x[1], reverse=True)
    return {
        "documents_dir": str(docs.relative_to(REPO_ROOT)),
        "files": [name for name, _ in items[:100]],
    }


def _document_path_under_docs(name: str) -> Path:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "invalid filename")
    p = (DEFAULT_DOCS / name).resolve()
    root = DEFAULT_DOCS.resolve()
    if not str(p).startswith(str(root)) or not p.is_file():
        raise HTTPException(404, "text file not found")
    if p.suffix.lower() != ".txt" or "ref_voice" in p.name.lower():
        raise HTTPException(403, "file not allowed")
    return p


DOC_TEXT_PREVIEW_MAX = 100_000


@app.get("/api/document-text")
def get_document_text(
    name: str = Query(..., description="Basename under documents/, e.g. 01_intro.txt"),
) -> dict:
    p = _document_path_under_docs(name)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise HTTPException(500, str(e)) from e
    truncated = len(text) > DOC_TEXT_PREVIEW_MAX
    if truncated:
        text = text[:DOC_TEXT_PREVIEW_MAX]
    return {"name": name, "text": text, "truncated": truncated}


@app.get("/api/outputs")
def list_outputs() -> dict:
    """Recent .m4a files in outputs/qwen3_clone (newest first)."""
    out_dir = REPO_ROOT / "outputs" / "qwen3_clone"
    if not out_dir.is_dir():
        return {"output_dir": str(out_dir.relative_to(REPO_ROOT)), "files": []}
    files: list[dict[str, object]] = []
    for p in out_dir.glob("*.m4a"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        try:
            st = p.stat()
            files.append(
                {
                    "path": str(p.resolve()),
                    "name": p.name,
                    "mtime": st.st_mtime,
                }
            )
        except OSError:
            continue
    files.sort(key=lambda x: float(x["mtime"]), reverse=True)
    return {
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "files": files[:100],
    }


def _audio_file_under_repo(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    root = REPO_ROOT.resolve()
    if not str(p).startswith(str(root)):
        raise HTTPException(403, "path must be under project root")
    if not p.is_file():
        raise HTTPException(404, "file not found")
    return p


@app.get("/api/audio")
def serve_audio(path: str = Query(..., description="Absolute or project-relative output path")):
    p = _audio_file_under_repo(path)
    return FileResponse(p, media_type="audio/mp4" if p.suffix.lower() == ".m4a" else "audio/wav")


class SuggestSpeedBody(BaseModel):
    output_path: str
    target_seconds: float
    current_speed: float


class FitOutputBody(BaseModel):
    output_path: str
    target_seconds: float


@app.post("/api/suggest-speed")
def suggest_speed(body: SuggestSpeedBody) -> dict:
    """Compare existing file to a goal length; suggest next --speed (no synthesis)."""
    if body.target_seconds <= 0:
        raise HTTPException(400, "target_seconds must be positive")
    p = _audio_file_under_repo(body.output_path)
    dur = _ffprobe_duration(p)
    suggested = body.current_speed * (dur / body.target_seconds)
    suggested = max(0.5, min(2.5, suggested))
    return {
        "duration_s": dur,
        "target_s": body.target_seconds,
        "delta_s": dur - body.target_seconds,
        "suggested_speed": suggested,
    }


@app.post("/api/fit-output")
def fit_output(body: FitOutputBody) -> dict:
    """Time-stretch an existing output to match target duration (ffmpeg)."""
    if body.target_seconds <= 0:
        raise HTTPException(400, "target_seconds must be positive")
    src = _audio_file_under_repo(body.output_path)
    dst = src.with_stem(src.stem + "_fit")
    try:
        _ffmpeg_fit_duration(src, body.target_seconds, dst)
    except Exception as e:
        raise HTTPException(500, f"ffmpeg: {e}") from e
    return {"output_path": str(dst)}


def _build_qwen_cmd(body: GenerateBody) -> list[str]:
    ref = _default_ref_audio()
    if ref is None:
        raise HTTPException(400, "Place securities_clip.m4a in the project root or extend the UI for ref audio.")
    tf = (DEFAULT_DOCS / body.text_file).resolve()
    if not str(tf).startswith(str(DEFAULT_DOCS.resolve())) or not tf.is_file():
        raise HTTPException(400, f"text file not found: {body.text_file}")

    cmd: list[str | Path] = [
        "uv",
        "run",
        "python",
        "-u",
        str(QWEN_SCRIPT),
        "--ref-audio",
        str(ref),
        "--ref-text-file",
        str(DEFAULT_REF_TEXT),
        "--text-file",
        str(tf),
        "--quality",
        body.quality,
        "--chunk-chars",
        str(body.chunk_chars),
        "--temperature",
        str(body.temperature),
        "--max-tokens",
        str(body.max_tokens),
        "--speed",
        str(body.speed),
        "--output-format",
        "m4a",
        "--aac-bitrate",
        "192k",
    ]
    if body.fast:
        cmd.append("--fast")
    return [str(x) for x in cmd]


def _stream_generate(body: GenerateBody):
    cmd = _build_qwen_cmd(body)
    yield f"data: {json.dumps({'log': ' '.join(cmd[:8]) + ' ...\\n'})}\n\n"

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    last_line = ""
    for line in proc.stdout:
        last_line = line.strip() if line.strip() else last_line
        yield f"data: {json.dumps({'log': line})}\n\n"

    code = proc.wait()
    if code != 0:
        yield f"data: {json.dumps({'error': f'qwen3_clone exited {code}'})}\n\n"
        return

    out_path = Path(last_line) if last_line else None
    if out_path is None or not out_path.is_file():
        yield f"data: {json.dumps({'error': 'no output path from generator'})}\n\n"
        return

    fit_applied = False
    if body.fit_ffmpeg and body.target_seconds is not None and body.target_seconds > 0:
        dst = out_path.with_stem(out_path.stem + "_fit")
        try:
            _ffmpeg_fit_duration(out_path, body.target_seconds, dst)
            out_path = dst
            fit_applied = True
        except Exception as e:
            yield f"data: {json.dumps({'error': f'ffmpeg fit failed: {e}'})}\n\n"
            return

    duration_s: float | None = None
    try:
        duration_s = _ffprobe_duration(out_path)
    except Exception:
        pass

    suggest: dict[str, float] | None = None
    if (
        duration_s is not None
        and not fit_applied
        and body.target_seconds is not None
        and body.target_seconds > 0
    ):
        tgt = body.target_seconds
        suggested = body.speed * (duration_s / tgt)
        suggested = max(0.5, min(2.5, suggested))
        suggest = {
            "duration_s": duration_s,
            "target_s": tgt,
            "delta_s": duration_s - tgt,
            "suggested_speed": suggested,
        }

    yield f"data: {json.dumps({'done': True, 'output_path': str(out_path), 'fit_applied': fit_applied, 'duration_s': duration_s, 'suggest': suggest})}\n\n"


@app.post("/api/generate")
def generate(body: GenerateBody):
    if body.fit_ffmpeg and (body.target_seconds is None or body.target_seconds <= 0):
        raise HTTPException(400, "ffmpeg fit requires a positive target length")
    return StreamingResponse(_stream_generate(body), media_type="text/event-stream")


def run() -> None:
    import uvicorn

    if not QWEN_SCRIPT.is_file():
        print(f"error: missing {QWEN_SCRIPT}", file=sys.stderr)
        raise SystemExit(2)
    host = "127.0.0.1"
    port = 8765
    print(f"Open http://{host}:{port}/  (repo: {REPO_ROOT})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
