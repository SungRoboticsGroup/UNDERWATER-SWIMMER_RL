#!/usr/bin/env python3
"""Convert a GIF to an MP4 video using Pillow + OpenCV (no ffmpeg needed).

Usage:
    python gif2vid.py INPUT.gif [OUTPUT.mp4] [--fps N]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence


def gif_fps(img: Image.Image) -> float:
    durations_ms = [frame.info.get("duration", 100) for frame in ImageSequence.Iterator(img)]
    avg_ms = sum(durations_ms) / max(len(durations_ms), 1)
    return 1000.0 / avg_ms if avg_ms > 0 else 10.0


def convert(input_gif: Path, output_mp4: Path, fps: float | None) -> bool:
    img = Image.open(input_gif)

    if fps is None:
        fps = gif_fps(img)

    frames = []
    for frame in ImageSequence.Iterator(img):
        rgb = frame.convert("RGB")
        frames.append(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))

    if not frames:
        print("No frames found in GIF", file=sys.stderr)
        return False

    h, w = frames[0].shape[:2]
    # mp4v: widely supported, no ffmpeg required (uses OpenCV's bundled codec).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Failed to open VideoWriter for {output_mp4}", file=sys.stderr)
        return False

    for f in frames:
        writer.write(f)
    writer.release()

    size_mb = output_mp4.stat().st_size / (1024 * 1024)
    print(f"✓ Wrote {output_mp4} ({len(frames)} frames @ {fps:.1f} fps, {size_mb:.2f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a GIF to MP4 (Pillow + OpenCV).")
    parser.add_argument("input", type=Path, help="Path to input .gif")
    parser.add_argument("output", type=Path, nargs="?", help="Path to output .mp4 (default: same name)")
    parser.add_argument("--fps", type=float, default=None, help="Override fps (default: derive from GIF)")
    args = parser.parse_args()

    input_gif: Path = args.input
    if not input_gif.is_file():
        print(f"Input not found: {input_gif}", file=sys.stderr)
        return 1

    output_mp4: Path = args.output if args.output is not None else input_gif.with_suffix(".mp4")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    return 0 if convert(input_gif, output_mp4, args.fps) else 1


if __name__ == "__main__":
    sys.exit(main())
