#!/usr/bin/env python3
"""
Minimal Qwen3-TTS voice clone on Apple Silicon (MLX).

Requires: macOS + Apple Silicon, ffmpeg on PATH (ref convert + default .m4a export),
network on first run (Hugging Face model cache).

Example:
  uv run python scripts/qwen3_clone.py \\
    --ref-audio ./samples/me.wav \\
    --ref-text-file documents/ref_transcript.txt \\
    --text-file documents/my_script.txt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import List

# MLX initializes Metal at import time; defer until after argparse --help.
SAMPLE_RATE = 24_000

MODEL_LITE = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
MODEL_PRO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"


def _ref_cache_key(src: Path) -> str:
    st = src.stat()
    raw = f"{src.resolve().as_posix()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _ensure_wav(path: Path, *, cache_dir: Path | None) -> Path:
    """Return path to 24kHz mono WAV, converting with ffmpeg if needed.

    If cache_dir is provided, reuse a cached conversion keyed by the source
    file path + size + mtime so repeated runs avoid re-encoding.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == SAMPLE_RATE:
                    return path
        except wave.Error:
            pass

    out: Path
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / f"ref_{_ref_cache_key(path)}_{SAMPLE_RATE}hz_mono.wav"
        if out.is_file():
            return out
    else:
        out = Path.cwd() / f"_qwen3_clone_convert_{int(time.time())}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed: {e.stderr or e.stdout or e}"
        ) from e
    return out


def _run(
    *,
    model: object,
    text: str,
    ref_audio: Path,
    ref_text: str | None,
    output_dir: Path,
    file_prefix: str,
    voice: str,
    speed: float,
    max_tokens: int,
    temperature: float,
    quiet: bool,
    stt_model: str | None,
) -> Path:
    from mlx_audio.tts.generate import generate_audio

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / f"_gen_{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    generate_audio(
        text=text,
        model=model,
        voice=voice,
        speed=speed,
        max_tokens=max_tokens,
            temperature=temperature,
        ref_audio=str(ref_audio),
        ref_text=ref_text,
        output_path=str(tmp_dir),
        file_prefix=file_prefix,
        audio_format="wav",
        verbose=not quiet,
        stt_model=stt_model,
    )

    # mlx-audio writes e.g. {prefix}_000.wav under output_path
    wavs = sorted(tmp_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError("Generation finished but no WAV was written.")
    out_wav = wavs[0]

    final = output_dir / out_wav.name
    shutil.move(str(out_wav), final)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return final


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """Split long text into chunk-sized blocks, preferring paragraph/sentence boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            flush()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            sub = ""
            for sent in sentences:
                candidate = f"{sub} {sent}".strip()
                if sub and len(candidate) > max_chars:
                    chunks.append(sub.strip())
                    sub = sent
                else:
                    sub = candidate
            if sub:
                if len(sub) <= max_chars:
                    chunks.append(sub)
                else:
                    for i in range(0, len(sub), max_chars):
                        chunks.append(sub[i : i + max_chars].strip())
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > max_chars:
            flush()
            current = para
        else:
            current = candidate

    flush()
    return chunks or [text.strip()]


def _wav_duration_seconds(path: Path) -> float:
    """Duration of a PCM WAV in seconds (uses stdlib wave)."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def _media_duration_seconds(path: Path) -> float:
    """Duration in seconds: fast path for WAV, else ffprobe (m4a, mp3, …)."""
    if path.suffix.lower() == ".wav":
        try:
            return _wav_duration_seconds(path)
        except (OSError, wave.Error):
            pass
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(r.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "ffprobe failed (install ffmpeg: brew install ffmpeg). "
            f"{e}"
        ) from e


def _wav_to_m4a(wav_in: Path, m4a_out: Path, *, bitrate: str) -> Path:
    """Encode mono/stereo WAV to AAC in an .m4a container."""
    m4a_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(wav_in),
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        str(m4a_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg AAC encode failed: {e.stderr or e.stdout or e}"
        ) from e
    return m4a_out


def _format_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    s = int(seconds)
    ms = int(round((seconds - s) * 1000))
    if ms >= 1000:
        s += 1
        ms = 0
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}.{ms:03d}"
    return f"{m:d}:{sec:02d}.{ms:03d}"


def _print_output_stats(
    path: Path,
    *,
    speed: float,
    target_seconds: float | None,
    num_segments: int,
    generation_wall_s: float,
) -> None:
    """Playback length vs generation time (stderr). Stdout stays the output path only."""
    try:
        dur = _media_duration_seconds(path)
    except RuntimeError as e:
        print(f"Stats: could not read duration: {e}", file=sys.stderr)
        return

    seg = (
        f"{num_segments} segment(s) merged"
        if num_segments > 1
        else "single pass"
    )
    print(
        f"Stats: playback length (how long it sounds): {_format_hms(dur)} ({dur:.2f} s) | "
        f"{seg} | --speed {speed:g}x",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Stats: generation time (wall clock on this Mac, model+synth+export): "
        f"{generation_wall_s:.2f} s — not the same as playback length above",
        file=sys.stderr,
        flush=True,
    )

    if target_seconds is None or target_seconds <= 0:
        _print_summary_footer(dur, generation_wall_s)
        return

    delta = dur - target_seconds
    if abs(delta) < 0.5:
        print(
            f"Target: {_format_hms(target_seconds)} ({target_seconds:.1f} s) — "
            "within ~0.5 s of desired length.",
            file=sys.stderr,
            flush=True,
        )
        _print_summary_footer(dur, generation_wall_s)
        return

    if delta > 0:
        adj = "longer"
        hint = "increase --speed to shorten playback"
    else:
        adj = "shorter"
        hint = "decrease --speed to lengthen playback"

    # Rough model: duration ~ inversely proportional to speaking speed.
    suggested = speed * (dur / target_seconds)
    suggested = max(0.5, min(2.5, suggested))

    print(
        f"Target: {_format_hms(target_seconds)} ({target_seconds:.1f} s) — "
        f"output is {adj} by {abs(delta):.1f} s ({hint}). "
        f"Rough next try: --speed {suggested:.3g}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "Pitch change without re-synthesis: e.g. https://vocalremover.org/pitch",
        file=sys.stderr,
        flush=True,
    )
    _print_summary_footer(dur, generation_wall_s)


def _print_summary_footer(dur: float, generation_wall_s: float) -> None:
    print("---", file=sys.stderr, flush=True)
    print(
        f"SUMMARY: playback ~{_format_hms(dur)} ({dur:.1f}s) | generation ~{generation_wall_s:.0f}s wall time",
        file=sys.stderr,
        flush=True,
    )


def _configure_stdio() -> None:
    """Reduce cross-stream reordering when C extensions buffer stdout."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except (AttributeError, OSError, ValueError):
            pass


def _concat_wavs(parts: List[Path], final_path: Path) -> Path:
    """Concatenate WAV parts losslessly via ffmpeg concat demuxer."""
    manifest = final_path.with_suffix(".concat.txt")
    try:
        manifest.write_text(
            "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-c",
            "copy",
            str(final_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg concat failed: {e.stderr or e.stdout or e}") from e
    finally:
        manifest.unlink(missing_ok=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal MLX Qwen3-TTS voice cloning (local, Apple Silicon)."
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        required=True,
        help="Path to reference clip (wav, m4a, mp3, …).",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Exact transcript of the reference (best quality). "
        "If omitted, mlx-audio transcribes with Whisper (extra model download + slower).",
    )
    parser.add_argument(
        "--ref-text-file",
        type=Path,
        default=None,
        help="Read reference transcript from a UTF-8 file (overrides --ref-text).",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to speak (inline). Ignored if --text-file is set.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="UTF-8 file of text to speak (for long / multi-document scripts).",
    )
    parser.add_argument(
        "--quality",
        choices=("lite", "pro"),
        default="lite",
        help="lite=0.6B (faster), pro=1.7B (better). Default: lite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen3_clone"),
        help="Directory for generated audio (default: outputs/qwen3_clone).",
    )
    parser.add_argument(
        "--no-ref-cache",
        action="store_true",
        help="Disable caching of converted reference WAV (forces ffmpeg each run).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Vivian",
        help="Qwen3 speaker name (e.g. Vivian, Chelsie). Default: Vivian.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.08,
        help="Speaking rate (1.0 = model default, >1 faster). Default: 1.08 (~8%% quicker).",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo-asr-fp16",
        help="Whisper repo when --ref-text is omitted (mlx-audio default).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output from the generator.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play result with afplay (macOS).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Token cap for generation (mlx-audio default 1200 truncates long scripts).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=700,
        help="Chunk long text into this many chars per pass. Larger = fewer chunks.",
    )
    parser.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable chunking and generate the whole text in one pass.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fewer ICL passes: enforce a higher chunk floor (~900 chars) for voice-clone mode "
        "(each chunk runs a full reference encode; fewer chunks = faster).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="Lower values improve consistency on long scripts (default: 0.45).",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=None,
        help="Optional. After the run, compare playback to this length (seconds) and print a suggested "
        "--speed for the next run. Does not change this output. Omit it to only print duration + "
        "adjust speed yourself.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Do not print duration / target comparison (stdout is still the output file path).",
    )
    parser.add_argument(
        "--output-format",
        choices=("m4a", "wav"),
        default="m4a",
        help="Final file format (default: m4a AAC via ffmpeg). Intermediate synthesis is always WAV.",
    )
    parser.add_argument(
        "--aac-bitrate",
        type=str,
        default="192k",
        help="AAC bitrate when --output-format m4a (default: 192k).",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="When output is m4a, also keep the intermediate .wav next to it.",
    )
    args = parser.parse_args()
    _configure_stdio()

    name_slug: str | None = None
    if args.text_file is not None:
        tf = args.text_file.expanduser().resolve()
        if not tf.is_file():
            parser.error(f"--text-file not found: {tf}")
        text = tf.read_text(encoding="utf-8").strip()
        name_slug = tf.stem
    elif args.text is not None:
        text = args.text.strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.error("Provide text via --text, --text-file, or pipe stdin.")

    if not text:
        parser.error("No text to synthesize (empty).")

    ref_text: str | None = args.ref_text
    if args.ref_text_file is not None:
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()

    model_id = MODEL_PRO if args.quality == "pro" else MODEL_LITE
    cache_dir = None if args.no_ref_cache else (args.output_dir / "_cache")
    ref_wav = _ensure_wav(args.ref_audio, cache_dir=cache_dir)
    stt_model: str | None = args.stt_model if ref_text is None else None

    if name_slug:
        slug = re.sub(r"[^\w\s-]", "", name_slug)[:60].strip().replace(" ", "_") or "speech"
    else:
        slug = re.sub(r"[^\w\s-]", "", text)[:40].strip().replace(" ", "_") or "out"
    file_prefix = f"{time.strftime('%H%M%S')}_{slug}"

    path: Path | None = None
    num_segments = 1
    t_gen_start = time.perf_counter()
    try:
        from mlx_audio.tts.utils import load_model

        # Load once so chunked runs do not repeatedly fetch/initialize.
        loaded_model = load_model(model_id)

        if args.no_chunk:
            chunks = [text]
        else:
            cc = max(200, args.chunk_chars)
            if args.fast:
                cc = max(cc, 900)
            chunks = _split_long_text(text, max_chars=cc)
        num_segments = len(chunks)

        if len(chunks) == 1:
            path = _run(
                model=loaded_model,
                text=chunks[0],
                ref_audio=ref_wav,
                ref_text=ref_text,
                output_dir=args.output_dir,
                file_prefix=file_prefix,
                voice=args.voice,
                speed=args.speed,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                quiet=args.quiet,
                stt_model=stt_model,
            )
        else:
            part_paths: List[Path] = []
            for i, chunk in enumerate(chunks, start=1):
                part_prefix = f"{file_prefix}_part{i:02d}"
                if not args.quiet:
                    print(f"Generating chunk {i}/{len(chunks)}...")
                part_paths.append(
                    _run(
                        model=loaded_model,
                        text=chunk,
                        ref_audio=ref_wav,
                        ref_text=ref_text,
                        output_dir=args.output_dir,
                        file_prefix=part_prefix,
                        voice=args.voice,
                        speed=args.speed,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        quiet=args.quiet,
                        stt_model=stt_model,
                    )
                )
            merged = args.output_dir / f"{file_prefix}.wav"
            path = _concat_wavs(part_paths, merged)
            for p in part_paths:
                p.unlink(missing_ok=True)
    finally:
        if ref_wav.name.startswith("_qwen3_clone_convert_"):
            ref_wav.unlink(missing_ok=True)

    if path is not None and args.output_format == "m4a":
        m4a_path = path.with_suffix(".m4a")
        _wav_to_m4a(path, m4a_path, bitrate=args.aac_bitrate)
        if not args.keep_wav:
            path.unlink(missing_ok=True)
        path = m4a_path

    generation_wall_s = time.perf_counter() - t_gen_start

    if path is not None:
        if not args.no_stats:
            sys.stdout.flush()
            print(
                "Whole-file summary (mlx Duration / Processing time lines are per chunk):",
                file=sys.stderr,
                flush=True,
            )
            _print_output_stats(
                path,
                speed=args.speed,
                target_seconds=args.target_seconds,
                num_segments=num_segments,
                generation_wall_s=generation_wall_s,
            )
        print(path, flush=True)
    if args.play and path is not None:
        subprocess.run(["afplay", str(path)], check=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
