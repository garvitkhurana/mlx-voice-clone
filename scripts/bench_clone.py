#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path


def _default_ref_audio() -> Path | None:
    local = Path("sample_clip.m4a")
    return local if local.is_file() else None


def _iter_text_files(documents_dir: Path) -> list[Path]:
    files = sorted(documents_dir.glob("0*.txt"))
    return [p for p in files if "ref_voice" not in p.name]


def _run_live(cmd: list[str]) -> tuple[int, str]:
    """Stream stdout/stderr live; return exit code + last non-empty stdout line (output path)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None

    last_stdout = ""

    def drain_stdout() -> None:
        nonlocal last_stdout
        for line in proc.stdout:
            sys.stdout.write(line)
            if line.strip():
                last_stdout = line.strip()

    def drain_stderr() -> None:
        for line in proc.stderr:
            sys.stderr.write(line)

    t_out = threading.Thread(target=drain_stdout)
    t_err = threading.Thread(target=drain_stderr)
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    proc.wait()
    return proc.returncode, last_stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark runner for scripts/qwen3_clone.py (timed, repeatable)."
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=_default_ref_audio(),
        help="Reference audio (default: ./sample_clip.m4a if it exists).",
    )
    parser.add_argument(
        "--ref-text-file",
        type=Path,
        default=Path("documents/ref_voice_transcript.txt"),
        help="Reference transcript file (default: documents/ref_voice_transcript.txt).",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=Path("documents"),
        help="Directory containing 0*.txt files to synthesize (default: documents).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only run a single text file by stem (e.g. 01_intro).",
    )
    parser.add_argument("--quality", choices=("lite", "pro"), default="lite")
    parser.add_argument("--chunk-chars", type=int, default=700)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Forward to qwen3_clone: larger chunk floor for fewer ICL passes (faster).",
    )
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--speed", type=float, default=1.08)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to the generator (less console noise).",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=None,
        help="Optional. Forward to qwen3_clone: after run, compare to this length and suggest next --speed.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Forward to qwen3_clone: suppress duration / target lines.",
    )
    parser.add_argument(
        "--output-format",
        choices=("m4a", "wav"),
        default="m4a",
        help="Forward to qwen3_clone (default: m4a).",
    )
    parser.add_argument(
        "--aac-bitrate",
        type=str,
        default="192k",
        help="Forward to qwen3_clone when output is m4a.",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Forward to qwen3_clone: keep intermediate WAV next to m4a.",
    )
    args = parser.parse_args()

    if args.ref_audio is None:
        print("error: --ref-audio is required (default not found).", file=sys.stderr)
        raise SystemExit(2)

    repo_root = Path.cwd()
    qwen_script = repo_root / "scripts" / "qwen3_clone.py"
    if not qwen_script.is_file():
        print(f"error: not found: {qwen_script}", file=sys.stderr)
        raise SystemExit(2)

    docs_dir = args.documents_dir
    if not docs_dir.is_dir():
        print(f"error: --documents-dir not found: {docs_dir}", file=sys.stderr)
        raise SystemExit(2)

    files = _iter_text_files(docs_dir)
    if args.only:
        files = [p for p in files if p.stem == args.only]
    if not files:
        print("error: no matching text files found.", file=sys.stderr)
        raise SystemExit(2)

    total_start = time.perf_counter()
    for tf in files:
        cmd = [
            "uv",
            "run",
            "python",
            "-u",
            str(qwen_script),
            "--ref-audio",
            str(args.ref_audio),
            "--ref-text-file",
            str(args.ref_text_file),
            "--text-file",
            str(tf),
            "--quality",
            args.quality,
            "--chunk-chars",
            str(args.chunk_chars),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--speed",
            str(args.speed),
        ]
        if args.quiet:
            cmd.append("--quiet")
        if args.fast:
            cmd.append("--fast")
        if args.target_seconds is not None:
            cmd.extend(["--target-seconds", str(args.target_seconds)])
        if args.no_stats:
            cmd.append("--no-stats")
        cmd.extend(["--output-format", args.output_format])
        if args.output_format == "m4a":
            cmd.extend(["--aac-bitrate", args.aac_bitrate])
        if args.keep_wav:
            cmd.append("--keep-wav")

        start = time.perf_counter()
        rc, out_path = _run_live(cmd)
        elapsed = time.perf_counter() - start

        if rc != 0:
            print(
                f"\nerror: failed on {tf.name} (exit {rc})",
                file=sys.stderr,
            )
            raise SystemExit(rc)

        # qwen3_clone: last stdout line is the output file path.
        out_p = Path(out_path)
        playback_s: float | None = None
        if out_p.is_file():
            try:
                r = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(out_p),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                playback_s = float(r.stdout.strip())
            except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
                playback_s = None

        lines = [
            "",
            "=" * 72,
            f"  BENCH SUMMARY — {tf.name}",
            f"  output file: {out_path}",
            f"  generation (wall): {elapsed:.2f}s  (time to run this job on your Mac)",
        ]
        if playback_s is not None:
            m, s = divmod(int(playback_s), 60)
            lines.append(
                f"  playback length: {m}m {s}s ({playback_s:.2f}s)  (how long the audio plays)"
            )
        lines.extend(
            [
                "  mlx log lines above: Duration / Processing time are per chunk, not the whole file.",
                "=" * 72,
                "",
            ]
        )
        print("\n".join(lines), flush=True)

    total_elapsed = time.perf_counter() - total_start
    print(
        f"batch generation (wall): {total_elapsed:.2f}s for {len(files)} file(s)",
        flush=True,
    )


if __name__ == "__main__":
    main()

