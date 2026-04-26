#!/usr/bin/env python3
"""
Standalone bandwidth simulation — no cv2 or ultralytics required.

Generates synthetic 640x480 frames, JPEG-encodes them with PIL to measure
real compressed sizes, then prints the bandwidth comparison table for
paper Section 4.5.

Usage:
    python bandwidth_sim.py [--frames 100] [--fps 30] [--quality 80]
"""

import argparse
import io
import random
import statistics
import struct
import time

from PIL import Image
import numpy as np


EDGE_BYTES_PER_CYCLE = 500      # MQTT cycle result payload (Section 2.4)
ASSUMED_CYCLE_TIME_S = 60.0     # typical assembly cycle length


def make_synthetic_frame(i: int, total: int) -> np.ndarray:
    """640x480 gradient + noise + moving rectangle — realistic scene complexity."""
    rng = np.random.default_rng(seed=i)
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(30, 80, w, dtype=np.uint8)
    frame[:, :, 1] = np.linspace(20, 60, w, dtype=np.uint8)
    frame[:, :, 2] = np.linspace(10, 40, w, dtype=np.uint8)
    frame = np.clip(
        frame.astype(np.int16) + rng.integers(-15, 15, frame.shape, dtype=np.int16),
        0, 255,
    ).astype(np.uint8)
    # moving white rectangle simulating hand/arm
    x = int(w * 0.1 + w * 0.6 * (i / max(total - 1, 1)))
    frame[180:300, x:x+60] = 200
    return frame


def jpeg_encode(frame_array: np.ndarray, quality: int) -> bytes:
    img = Image.fromarray(frame_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def run(num_frames: int, fps: int, quality: int) -> None:
    print(f"Encoding {num_frames} synthetic 640x480 frames at JPEG quality={quality}...")
    t0 = time.perf_counter()

    jpeg_sizes: list[int] = []
    for i in range(num_frames):
        frame = make_synthetic_frame(i, num_frames)
        jpg = jpeg_encode(frame, quality)
        jpeg_sizes.append(len(jpg))

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s.\n")

    # --- stats ---
    avg_bytes   = statistics.mean(jpeg_sizes)
    min_bytes   = min(jpeg_sizes)
    max_bytes   = max(jpeg_sizes)
    p95_bytes   = sorted(jpeg_sizes)[int(len(jpeg_sizes) * 0.95)]

    streaming_bps        = avg_bytes * fps                      # bytes/s
    streaming_mbps       = streaming_bps / 1_000_000           # MB/s
    streaming_gb_per_hr  = streaming_mbps * 3600 / 1000        # GB/hr
    cloud_mb_per_cycle   = streaming_mbps * ASSUMED_CYCLE_TIME_S

    edge_bps             = EDGE_BYTES_PER_CYCLE / ASSUMED_CYCLE_TIME_S   # bytes/s
    edge_kb_per_hr       = edge_bps * 3600 / 1000              # KB/hr
    reduction            = streaming_bps / edge_bps

    # --- print tables ---
    print("=" * 62)
    print("JPEG FRAME SIZE (640x480, synthetic scene)")
    print("=" * 62)
    print(f"  {'Frames measured':<30} {num_frames:>10}")
    print(f"  {'Avg frame size':<30} {avg_bytes/1024:>9.1f} KB")
    print(f"  {'Min frame size':<30} {min_bytes/1024:>9.1f} KB")
    print(f"  {'Max frame size':<30} {max_bytes/1024:>9.1f} KB")
    print(f"  {'P95 frame size':<30} {p95_bytes/1024:>9.1f} KB")

    print()
    print("=" * 62)
    print("BANDWIDTH COMPARISON  (paper Section 4.5)")
    print("=" * 62)
    print(f"  {'Metric':<38} {'Edge':>10} {'Cloud':>10}")
    print("  " + "-" * 58)
    print(f"  {'Data per frame':<38} {'—':>10} {avg_bytes/1024:>8.1f} KB")
    print(f"  {'Streaming rate @ ' + str(fps) + ' fps':<38} {'—':>10} {streaming_mbps:>7.2f} MB/s")
    print(f"  {'Data per cycle (~60 s)':<38} {EDGE_BYTES_PER_CYCLE/1024:>9.2f} KB {cloud_mb_per_cycle:>7.0f} MB")
    print(f"  {'Data per hour (est.)':<38} {edge_kb_per_hr:>8.1f} KB {streaming_gb_per_hr:>8.1f} GB")
    print(f"  {'Bandwidth reduction':<38} {f'{reduction:,.0f}x':>10} {'—':>10}")
    print("=" * 62)

    print(f"""
Paper table values (copy-paste ready):
  Data per cycle  | Edge: ~{EDGE_BYTES_PER_CYCLE} B  | Cloud: ~{cloud_mb_per_cycle:.0f} MB
  Data per hour   | Edge: ~{edge_kb_per_hr:.0f} KB   | Cloud: ~{streaming_gb_per_hr:.1f} GB
  Reduction       | >{reduction:,.0f}x
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandwidth simulation (no cv2 required)")
    parser.add_argument("--frames",  type=int, default=100, help="Frames to sample (default: 100)")
    parser.add_argument("--fps",     type=int, default=30,  help="Assumed streaming FPS (default: 30)")
    parser.add_argument("--quality", type=int, default=80,  help="JPEG quality 1-95 (default: 80)")
    args = parser.parse_args()
    run(args.frames, args.fps, args.quality)


if __name__ == "__main__":
    main()