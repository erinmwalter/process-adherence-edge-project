#!/usr/bin/env python3
"""
Benchmark — measures edge vs cloud latency and bandwidth on the same frames.

Runs two measurement passes:
  1. EDGE:  capture frame → YOLO + zone check locally → record latency
  2. CLOUD: capture frame → JPEG encode → MQTT to cloud server → wait for
            MQTT response → record round-trip latency + bytes sent

Also computes bandwidth comparison matching paper Section 4.5:
  - Cloud path: JPEG streaming rate at 30 fps
  - Edge path: ~500-byte MQTT cycle result per cycle

Use --synthetic to generate frames without a camera (simulation mode).

Usage:
  # Terminal 1 — start the cloud sim server:
  python -m src.benchmark_server --mqtt-host localhost

  # Terminal 2 — run with real camera:
  python -m src.benchmark --mqtt-host localhost --frames 100

  # Run in simulation mode (no camera needed):
  python -m src.benchmark --synthetic --frames 100
"""

import argparse
import base64
import csv
import json
import os
import statistics
import sys
import threading
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from src.detector import Detector
from src.zone_manager import ZoneManager

TOPIC_FRAME = "benchmark/frame"
TOPIC_RESULT = "benchmark/result"


EDGE_BYTES_PER_CYCLE = 500      # ~500-byte MQTT cycle result (from paper Section 2.4)
VIDEO_FPS = 30                  # assumed streaming frame rate for bandwidth calc


class Benchmark:
    def __init__(
        self,
        camera_index: int,
        num_frames: int,
        broker_host: str,
        broker_port: int,
        config_path: str | None,
        yolo_model: str,
        confidence: float,
        output_csv: str,
        use_synthetic: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.num_frames = num_frames
        self.config_path = config_path
        self.output_csv = output_csv
        self.use_synthetic = use_synthetic

        # edge-side detector and zone manager
        self.detector = Detector(yolo_model=yolo_model, confidence=confidence)
        self.zone_manager = ZoneManager()

        # MQTT for cloud path
        self._client = mqtt.Client(protocol=mqtt.MQTTv311)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._broker_host = broker_host
        self._broker_port = broker_port

        # cloud round-trip tracking
        self._pending: dict[int, float] = {}   # frame_id → sent_at (perf_counter)
        self._cloud_results: dict[int, dict] = {}
        self._cloud_event = threading.Event()

        # bandwidth tracking
        self._total_cloud_bytes: int = 0

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_RESULT)

    def _on_message(self, client, userdata, msg):
        try:
            result = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        fid = result.get("frame_id", -1)
        if fid in self._pending:
            result["client_recv_at"] = time.perf_counter()
            result["round_trip_ms"] = (
                (result["client_recv_at"] - self._pending[fid]) * 1000
            )
            self._cloud_results[fid] = result
            self._cloud_event.set()

    # ── synthetic frame generation ───────────────────────────────

    def _generate_synthetic_frames(self) -> list[np.ndarray]:
        """Generate realistic-looking synthetic frames without a camera.

        Uses a gradient background with a moving white rectangle to give YOLO
        something to process (avoids trivially-fast inference on blank frames).
        """
        print(f"Generating {self.num_frames} synthetic frames (640x480)...")
        frames = []
        h, w = 480, 640
        rng = np.random.default_rng(seed=42)
        for i in range(self.num_frames):
            # gradient base
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = np.linspace(30, 80, w, dtype=np.uint8)
            frame[:, :, 1] = np.linspace(20, 60, w, dtype=np.uint8)
            frame[:, :, 2] = np.linspace(10, 40, w, dtype=np.uint8)
            # add mild noise
            frame = np.clip(
                frame.astype(np.int16) + rng.integers(-15, 15, frame.shape, dtype=np.int16),
                0, 255,
            ).astype(np.uint8)
            # moving rectangle simulating a hand/arm region
            x = int(w * 0.1 + (w * 0.6) * (i / max(self.num_frames - 1, 1)))
            cv2.rectangle(frame, (x, 180), (x + 60, 300), (200, 200, 200), -1)
            frames.append(frame)
        print(f"Generated {len(frames)} synthetic frames.\n")
        return frames

    # ── edge benchmark ───────────────────────────────────────────

    def _run_edge_pass(self, frames: list[np.ndarray]) -> list[float]:
        """Run full edge pipeline on each frame, return latencies in ms."""
        latencies = []
        for frame in frames:
            t0 = time.perf_counter()

            poses = self.detector.detect_poses(frame)
            if poses:
                pose = poses[0]
                wrist = pose.right_wrist or pose.left_wrist
                if wrist:
                    self.zone_manager.get_zones_for_point(wrist.x, wrist.y)

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

        return latencies

    # ── cloud benchmark ──────────────────────────────────────────

    def _run_cloud_pass(self, frames: list[np.ndarray]) -> list[float]:
        """Send each frame to cloud server via MQTT, measure round-trip."""
        latencies = []
        self._total_cloud_bytes = 0

        for i, frame in enumerate(frames):
            # JPEG encode
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_bytes = jpg.tobytes()
            self._total_cloud_bytes += len(jpg_bytes)
            b64 = base64.b64encode(jpg_bytes).decode("ascii")

            self._cloud_event.clear()
            sent_at = time.perf_counter()
            self._pending[i] = sent_at

            payload = json.dumps({
                "frame_id": i,
                "sent_at": sent_at,
                "frame_b64": b64,
            })
            self._client.publish(TOPIC_FRAME, payload)

            # wait for response (timeout 10s)
            if self._cloud_event.wait(timeout=10.0):
                result = self._cloud_results.get(i)
                if result:
                    latencies.append(result["round_trip_ms"])
                    continue

            # timeout
            print(f"  WARNING: Frame {i} timed out waiting for cloud response")
            latencies.append(float("nan"))

        return latencies

    # ── main ─────────────────────────────────────────────────────

    def run(self) -> None:
        # load zones
        try:
            if self.config_path:
                self.zone_manager.load(self.config_path)
            else:
                self.zone_manager.load()
            print(f"Loaded {len(self.zone_manager.zones)} zone(s)")
        except FileNotFoundError:
            print("WARNING: No zone config, zone checks will be empty")

        # capture or generate frames
        if self.use_synthetic:
            frames = self._generate_synthetic_frames()
        else:
            print(f"\nCapturing {self.num_frames} frames from camera {self.camera_index}...")
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                print(f"ERROR: Cannot open camera {self.camera_index}")
                sys.exit(1)

            frames: list[np.ndarray] = []
            for _ in range(self.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            print(f"Captured {len(frames)} frames.\n")

            if not frames:
                print("ERROR: No frames captured.")
                sys.exit(1)

        # --- pass 1: edge ---
        print(f"=== EDGE PASS ({len(frames)} frames) ===")
        edge_latencies = self._run_edge_pass(frames)
        print(f"  Done. Mean: {statistics.mean(edge_latencies):.1f}ms, "
              f"Median: {statistics.median(edge_latencies):.1f}ms\n")

        # --- pass 2: cloud ---
        print(f"=== CLOUD PASS ({len(frames)} frames) ===")
        print(f"  Connecting to broker at {self._broker_host}:{self._broker_port}...")
        try:
            self._client.connect(self._broker_host, self._broker_port, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            print(f"  ERROR: Cannot connect to broker: {e}")
            print("  Make sure the broker is running and benchmark_server is started.")
            print("  Skipping cloud pass.\n")
            self._print_results(edge_latencies, [])
            self._print_bandwidth([])
            return

        # small delay for subscription to propagate
        time.sleep(0.5)

        cloud_latencies = self._run_cloud_pass(frames)
        self._client.loop_stop()
        self._client.disconnect()

        valid_cloud = [x for x in cloud_latencies if not (x != x)]  # filter NaN
        if valid_cloud:
            print(f"  Done. Mean: {statistics.mean(valid_cloud):.1f}ms, "
                  f"Median: {statistics.median(valid_cloud):.1f}ms\n")
        else:
            print("  No valid cloud responses received.\n")

        self._print_results(edge_latencies, cloud_latencies)
        self._print_bandwidth(cloud_latencies)
        self._write_csv(edge_latencies, cloud_latencies)

    def _print_results(self, edge: list[float], cloud: list[float]) -> None:
        valid_cloud = [x for x in cloud if not (x != x)]

        print("=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"{'Metric':<30} {'Edge':>12} {'Cloud':>12}")
        print("-" * 60)

        if edge:
            print(f"{'Mean latency (ms)':<30} {statistics.mean(edge):>12.1f}", end="")
            if valid_cloud:
                print(f" {statistics.mean(valid_cloud):>12.1f}")
            else:
                print(f" {'N/A':>12}")

            print(f"{'Median latency (ms)':<30} {statistics.median(edge):>12.1f}", end="")
            if valid_cloud:
                print(f" {statistics.median(valid_cloud):>12.1f}")
            else:
                print(f" {'N/A':>12}")

            print(f"{'P95 latency (ms)':<30} {_percentile(edge, 95):>12.1f}", end="")
            if valid_cloud:
                print(f" {_percentile(valid_cloud, 95):>12.1f}")
            else:
                print(f" {'N/A':>12}")

            print(f"{'Min latency (ms)':<30} {min(edge):>12.1f}", end="")
            if valid_cloud:
                print(f" {min(valid_cloud):>12.1f}")
            else:
                print(f" {'N/A':>12}")

            print(f"{'Max latency (ms)':<30} {max(edge):>12.1f}", end="")
            if valid_cloud:
                print(f" {max(valid_cloud):>12.1f}")
            else:
                print(f" {'N/A':>12}")

            if valid_cloud:
                speedup = statistics.mean(valid_cloud) / statistics.mean(edge)
                print(f"\n{'Speedup (cloud/edge)':<30} {speedup:>12.1f}x")

        print("=" * 60)

    def _print_bandwidth(self, cloud: list[float]) -> None:
        """Print bandwidth comparison table matching paper Section 4.5."""
        valid_cloud = [x for x in cloud if not (x != x)]

        print("\n" + "=" * 60)
        print("BANDWIDTH COMPARISON (paper Section 4.5)")
        print("=" * 60)
        print(f"{'Metric':<38} {'Edge':>10} {'Cloud':>10}")
        print("-" * 60)

        # --- cloud streaming numbers ---
        if self._total_cloud_bytes > 0 and len(cloud) > 0:
            avg_jpeg_bytes = self._total_cloud_bytes / len(cloud)
            streaming_mbps = (avg_jpeg_bytes * VIDEO_FPS) / 1_000_000
            streaming_gb_per_hour = streaming_mbps * 3600 / 1000
            assumed_cycle_s = 60.0
            cloud_mb_per_cycle = streaming_mbps * assumed_cycle_s

            print(f"{'Avg JPEG frame size':<38} {'—':>10} {avg_jpeg_bytes/1024:>8.1f} KB")
            print(f"{'Streaming bitrate @ 30fps':<38} {'—':>10} {streaming_mbps:>7.2f} MB/s")
            print(f"{'Data per cycle (~60s)':<38} {EDGE_BYTES_PER_CYCLE/1024:>8.2f} KB {cloud_mb_per_cycle:>8.0f} MB")
            print(f"{'Data per hour (est.)':<38} {EDGE_BYTES_PER_CYCLE * 60 / 1e6:>7.2f} MB {streaming_gb_per_hour:>8.1f} GB")
            reduction = (streaming_mbps * 1_000_000) / (EDGE_BYTES_PER_CYCLE / assumed_cycle_s)
            print(f"{'Bandwidth reduction':<38} {f'{reduction:.0f}x':>10} {'—':>10}")
        else:
            # cloud pass was skipped — report edge-only numbers + theoretical cloud estimate
            # Use 640×480 JPEG at quality=80, typically ~15-25 KB per frame
            estimated_jpeg_kb = 20.0
            streaming_mbps = (estimated_jpeg_kb * 1024 * VIDEO_FPS) / 1_000_000
            streaming_gb_per_hour = streaming_mbps * 3600 / 1000
            assumed_cycle_s = 60.0
            cloud_mb_per_cycle = streaming_mbps * assumed_cycle_s
            reduction = (streaming_mbps * 1_000_000) / (EDGE_BYTES_PER_CYCLE / assumed_cycle_s)

            print(f"  (Cloud pass skipped — using estimated JPEG size of ~{estimated_jpeg_kb:.0f} KB)")
            print(f"{'Est. streaming bitrate @ 30fps':<38} {'—':>10} {streaming_mbps:>7.2f} MB/s")
            print(f"{'Data per cycle (~60s)':<38} {EDGE_BYTES_PER_CYCLE/1024:>8.2f} KB {cloud_mb_per_cycle:>8.0f} MB")
            print(f"{'Data per hour (est.)':<38} {EDGE_BYTES_PER_CYCLE * 60 / 1e6:>7.2f} MB {streaming_gb_per_hour:>8.1f} GB")
            print(f"{'Bandwidth reduction':<38} {f'>{reduction:.0f}x':>10} {'—':>10}")

        print("=" * 60)

    def _write_csv(self, edge: list[float], cloud: list[float]) -> None:
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "edge_ms", "cloud_ms"])
            for i in range(max(len(edge), len(cloud))):
                e = edge[i] if i < len(edge) else ""
                c = cloud[i] if i < len(cloud) else ""
                writer.writerow([i, e, c])
        print(f"\nPer-frame data written to {self.output_csv}")


def _percentile(data: list[float], pct: float) -> float:
    s = sorted(data)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark edge vs cloud inference latency"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to benchmark (default: 100)")
    parser.add_argument("--mqtt-host", type=str, default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to zones.json")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--output", type=str, default="benchmark_results.csv",
                        help="Output CSV path (default: benchmark_results.csv)")
    args = parser.parse_args()

    bench = Benchmark(
        camera_index=args.camera,
        num_frames=args.frames,
        broker_host=args.mqtt_host,
        broker_port=args.mqtt_port,
        config_path=args.config,
        yolo_model=args.model,
        confidence=args.conf,
        output_csv=args.output,
    )
    bench.run()


if __name__ == "__main__":
    main()
