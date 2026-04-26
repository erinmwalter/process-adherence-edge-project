"""
Run mode — live process-adherence monitoring.

Pipeline each frame:
  1. YOLOv8n-pose → extract operator wrist keypoints
  2. Check which zone each wrist is in
  3. Feed wrist position into SequenceTracker
  4. SequenceTracker compares observed picks vs expected trim sequence
  5. If deviation → trigger AlertSystem flash on screen
  6. Overlay: zones, wrist markers, sequence progress, parts, status panel
  7. On cycle end → publish CycleResult via MQTT

Controls in run mode:
  - Number keys 1-9 : manually set trim level (for testing without MQTT)
  - Enter            : end current cycle
  - 'n'              : start a new cycle with current trim
  - 'q' / Escape     : quit
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

from .zone_manager import ZoneManager
from .detector import Detector, PoseDetection, PartDetection
from .sequence_tracker import SequenceTracker, StepEvent
from .trim_config import TrimConfig
from .alert_system import AlertSystem
from .mqtt_client import MQTTClient, DummyMQTTClient

WINDOW_NAME = "Process Adherence — Run Mode"


class PipelineStats:
    """Collects per-frame timing breakdown for benchmarking."""

    def __init__(self) -> None:
        self.capture_ms: List[float] = []
        self.inference_ms: List[float] = []
        self.zone_check_ms: List[float] = []
        self.part_detect_ms: List[float] = []
        self.render_ms: List[float] = []
        self.total_ms: List[float] = []
        self.mqtt_payload_bytes: List[int] = []
        # accuracy tracking per cycle
        self.cycle_results: List[Dict] = []  # {correct_steps, total_steps, errors, completed}

    def record_cycle(self, expected: List[str], observed: List[str],
                     error_count: int, completed: bool) -> None:
        total_steps = len(observed)
        correct_steps = sum(
            1 for e, o in zip(expected, observed) if e == o
        )
        self.cycle_results.append({
            "correct_steps": correct_steps,
            "total_steps": total_steps,
            "expected_len": len(expected),
            "error_count": error_count,
            "completed": completed,
        })

    def record(self, capture: float, inference: float, zone_check: float,
               part_detect: float, render: float, total: float) -> None:
        self.capture_ms.append(capture)
        self.inference_ms.append(inference)
        self.zone_check_ms.append(zone_check)
        self.part_detect_ms.append(part_detect)
        self.render_ms.append(render)
        self.total_ms.append(total)

    def print_summary(self) -> None:
        import statistics
        import csv as _csv

        n = len(self.total_ms)
        if n == 0:
            print("\nNo frames recorded.")
            return

        def _stats(data: List[float]) -> Dict[str, float]:
            return {
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "p95": sorted(data)[min(int(n * 0.95), n - 1)],
                "min": min(data),
                "max": max(data),
            }

        stages = [
            ("Capture", self.capture_ms),
            ("YOLO Inference", self.inference_ms),
            ("Zone Check + Track", self.zone_check_ms),
            ("Part Detection", self.part_detect_ms),
            ("Render + Display", self.render_ms),
            ("TOTAL", self.total_ms),
        ]

        print(f"\n{'=' * 72}")
        print(f"PIPELINE LATENCY STATS ({n} frames)")
        print(f"{'=' * 72}")
        print(f"{'Stage':<22} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
        print(f"{'-' * 72}")
        for name, data in stages:
            s = _stats(data)
            print(f"{name:<22} {s['mean']:>7.1f}ms {s['median']:>7.1f}ms "
                  f"{s['p95']:>7.1f}ms {s['min']:>7.1f}ms {s['max']:>7.1f}ms")
        print(f"{'=' * 72}")

        effective_fps = 1000.0 / statistics.mean(self.total_ms) if statistics.mean(self.total_ms) > 0 else 0
        print(f"Effective FPS: {effective_fps:.1f}")

        # bandwidth stats
        if self.mqtt_payload_bytes:
            total_bytes = sum(self.mqtt_payload_bytes)
            num_cycles = len(self.mqtt_payload_bytes)
            avg_bytes = total_bytes / num_cycles
            print(f"\n{'=' * 72}")
            print(f"BANDWIDTH STATS ({num_cycles} cycles published)")
            print(f"{'=' * 72}")
            print(f"{'Avg payload size':<30} {avg_bytes:>10.0f} bytes")
            print(f"{'Min payload size':<30} {min(self.mqtt_payload_bytes):>10} bytes")
            print(f"{'Max payload size':<30} {max(self.mqtt_payload_bytes):>10} bytes")
            print(f"{'Total data sent':<30} {total_bytes:>10} bytes")
            # estimate cloud alternative: 640x480 JPEG @ 30fps @ ~30KB/frame
            run_seconds = n / effective_fps if effective_fps > 0 else 0
            cloud_bytes = run_seconds * 30 * 30_000  # 30fps × ~30KB/frame
            if cloud_bytes > 0:
                reduction = (1 - total_bytes / cloud_bytes) * 100
                print(f"{'Est. cloud video (same duration)':<30} {cloud_bytes / 1e6:>10.1f} MB")
                print(f"{'Bandwidth reduction':<30} {reduction:>9.1f}%")
            print(f"{'=' * 72}")

        # accuracy stats
        if self.cycle_results:
            num_cycles = len(self.cycle_results)
            total_steps = sum(c["total_steps"] for c in self.cycle_results)
            correct_steps = sum(c["correct_steps"] for c in self.cycle_results)
            total_errors = sum(c["error_count"] for c in self.cycle_results)
            completed_cycles = sum(1 for c in self.cycle_results if c["completed"])
            perfect_cycles = sum(1 for c in self.cycle_results if c["error_count"] == 0)

            step_accuracy = (correct_steps / total_steps * 100) if total_steps > 0 else 0
            cycle_pass_rate = (perfect_cycles / num_cycles * 100) if num_cycles > 0 else 0

            print(f"\n{'=' * 72}")
            print(f"ACCURACY STATS ({num_cycles} cycles)")
            print(f"{'=' * 72}")
            print(f"{'Total cycles':<30} {num_cycles:>10}")
            print(f"{'Completed cycles':<30} {completed_cycles:>10}")
            print(f"{'Perfect cycles (0 errors)':<30} {perfect_cycles:>10}")
            print(f"{'Cycle pass rate':<30} {cycle_pass_rate:>9.1f}%")
            print(f"{'Total pick steps':<30} {total_steps:>10}")
            print(f"{'Correct steps':<30} {correct_steps:>10}")
            print(f"{'Step-level accuracy':<30} {step_accuracy:>9.1f}%")
            print(f"{'Total deviations':<30} {total_errors:>10}")
            print(f"{'=' * 72}")

    def write_csv(self, path: str) -> None:
        import csv as _csv

        n = len(self.total_ms)
        with open(path, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow([
                "frame", "capture_ms", "inference_ms", "zone_check_ms",
                "part_detect_ms", "render_ms", "total_ms",
            ])
            for i in range(n):
                writer.writerow([
                    i,
                    round(self.capture_ms[i], 2),
                    round(self.inference_ms[i], 2),
                    round(self.zone_check_ms[i], 2),
                    round(self.part_detect_ms[i], 2),
                    round(self.render_ms[i], 2),
                    round(self.total_ms[i], 2),
                ])
        print(f"Per-frame stats written to {path}")

        # write accuracy CSV alongside frame stats
        if self.cycle_results:
            import os
            base, ext = os.path.splitext(path)
            acc_path = f"{base}_accuracy{ext}"
            with open(acc_path, "w", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow([
                    "cycle", "correct_steps", "total_steps", "expected_len",
                    "error_count", "completed",
                ])
                for i, c in enumerate(self.cycle_results):
                    writer.writerow([
                        i, c["correct_steps"], c["total_steps"],
                        c["expected_len"], c["error_count"], c["completed"],
                    ])
            print(f"Accuracy stats written to {acc_path}")


class RunMode:
    def __init__(
        self,
        camera_index: int = 0,
        config_path: str | None = None,
        trims_path: str | None = None,
        yolo_model: str = "yolov8n-pose.pt",
        confidence: float = 0.45,
        mqtt_host: str | None = None,
        mqtt_port: int = 1883,
        collect_stats: bool = False,
        stats_csv: str = "pipeline_stats.csv",
    ) -> None:
        self.camera_index = camera_index
        self.zone_manager = ZoneManager()
        self.detector = Detector(yolo_model=yolo_model, confidence=confidence)
        self.trim_config = TrimConfig()
        self.tracker = SequenceTracker(self.zone_manager)
        self.alert = AlertSystem()
        self.config_path = config_path
        self.trims_path = trims_path

        # stats collection
        self._collect_stats = collect_stats
        self._stats_csv = stats_csv
        self._stats = PipelineStats() if collect_stats else None

        # current trim (set via MQTT or keyboard)
        self._current_trim: str = ""
        self._trim_received_at: float = 0.0

        # MQTT
        if mqtt_host:
            self.mqtt: MQTTClient | DummyMQTTClient = MQTTClient(
                broker_host=mqtt_host, broker_port=mqtt_port,
            )
        else:
            self.mqtt = DummyMQTTClient()

    # ── MQTT trim callback ───────────────────────────────────────

    def _on_trim_received(self, trim_level: str) -> None:
        """Called when a trim message arrives via MQTT."""
        seq = self.trim_config.get_sequence(trim_level)
        if seq is None:
            print(f"WARNING: unknown trim '{trim_level}', ignoring.")
            return

        # if a cycle is already running, auto-end it and publish results
        if self.tracker.is_active:
            result = self.tracker.end_cycle()
            if result:
                print(f"\nAuto-ended previous cycle (trim={result.trim_level}, "
                      f"errors={result.error_count}, time={result.cycle_time:.1f}s)")
                self.mqtt.publish_cycle(result.to_dict())
                if self._stats is not None:
                    self._stats.record_cycle(
                        result.expected_sequence, result.observed_sequence,
                        result.error_count, result.completed,
                    )

        self._current_trim = trim_level
        self._trim_received_at = time.time()
        self.tracker.start_cycle(trim_level, seq)
        print(f"Cycle started — trim={trim_level}, sequence={seq}")

    # ── trim banner (big on-screen indicator) ────────────────────

    def _draw_trim_banner(self, frame: np.ndarray) -> None:
        """Draw a large banner at the top showing current trim or waiting state."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        banner_h = 60

        if self.tracker.is_active:
            # green banner — active cycle
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 120, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            trim_text = f"TRIM {self._current_trim}"
            seq_text = " > ".join(self.tracker.expected_sequence)
            step = self.tracker.current_step_index + 1
            total = len(self.tracker.expected_sequence)
            progress = f"Step {step}/{total}"

            cv2.putText(frame, trim_text, (15, 40),
                        font, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"{progress}  |  {seq_text}",
                        (250, 40), font, 0.6, (200, 255, 200), 1)
        else:
            # grey banner — waiting
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (60, 60, 60), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "WAITING FOR NEXT VEHICLE...", (15, 40),
                        font, 0.9, (150, 150, 150), 2)

    # ── status panel ─────────────────────────────────────────────

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        fps: float,
        zone_parts: Dict[str, List[PartDetection]],
    ) -> None:
        h, w = frame.shape[:2]
        panel_w = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        x0 = w - panel_w + 10
        y = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y), font, 0.55, (0, 255, 0), 1)
        y += 25

        # MQTT status
        mqtt_status = "CONNECTED" if self.mqtt.is_connected else "OFFLINE"
        mqtt_color = (0, 255, 0) if self.mqtt.is_connected else (0, 140, 255)
        cv2.putText(frame, f"MQTT: {mqtt_status}", (x0, y), font, 0.45, mqtt_color, 1)
        y += 25

        # Trim & cycle
        cv2.putText(frame, f"Trim: {self._current_trim or '(none)'}", (x0, y),
                     font, 0.55, (255, 255, 255), 1)
        y += 22
        cv2.putText(frame, f"Cycle: {'ACTIVE' if self.tracker.is_active else 'IDLE'}",
                     (x0, y), font, 0.5, (200, 200, 200), 1)
        y += 25

        # Expected sequence
        if self.tracker.is_active:
            cv2.putText(frame, "Expected:", (x0, y), font, 0.45, (180, 180, 180), 1)
            y += 18
            for i, zone_name in enumerate(self.tracker.expected_sequence):
                marker = ">" if i == self.tracker.current_step_index else " "
                done = i < self.tracker.current_step_index
                color = (0, 255, 0) if done else (200, 200, 200)
                if i == self.tracker.current_step_index:
                    color = (0, 255, 255)
                cv2.putText(frame, f" {marker} {i+1}. {zone_name}",
                             (x0, y), font, 0.45, color, 1)
                y += 18

            y += 8
            cv2.putText(frame, "Observed:", (x0, y), font, 0.45, (180, 180, 180), 1)
            y += 18
            for i, zone_name in enumerate(self.tracker.observed_sequence):
                expected = self.tracker.expected_sequence[i] if i < len(self.tracker.expected_sequence) else "?"
                ok = zone_name == expected
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(frame, f"  {i+1}. {zone_name}", (x0, y), font, 0.45, color, 1)
                y += 18

            y += 8
            cv2.putText(frame, f"Errors: {self.tracker.error_count}",
                         (x0, y), font, 0.5,
                         (0, 0, 255) if self.tracker.error_count else (0, 255, 0), 1)
            y += 25

        # Part inventory per zone
        for zone_name in sorted(zone_parts.keys()):
            parts = zone_parts[zone_name]
            if not parts:
                continue
            cv2.putText(frame, f"Zone {zone_name} parts:", (x0, y),
                         font, 0.45, (255, 255, 255), 1)
            y += 18
            color_counts: Dict[str, int] = defaultdict(int)
            for p in parts:
                color_counts[p.color_name] += 1
            for cname, cnt in sorted(color_counts.items()):
                cv2.putText(frame, f"  {cname}: {cnt}", (x0, y),
                             font, 0.4, (200, 200, 200), 1)
                y += 16
            y += 8

        # Keyboard help
        y = h - 60
        cv2.putText(frame, "1-9: set trim | N: new cycle", (x0, y),
                     font, 0.35, (140, 140, 140), 1)
        y += 16
        cv2.putText(frame, "ENTER: end cycle | Q: quit", (x0, y),
                     font, 0.35, (140, 140, 140), 1)

    # ── main loop ────────────────────────────────────────────────

    def run(self) -> None:
        # Load zone config
        try:
            if self.config_path:
                self.zone_manager.load(self.config_path)
            else:
                self.zone_manager.load()
        except FileNotFoundError:
            print("ERROR: No zone configuration found. Run 'configure' mode first.")
            return

        if not self.zone_manager.zones:
            print("ERROR: Zone config is empty. Run 'configure' mode first.")
            return
        print(f"Loaded {len(self.zone_manager.zones)} zone(s).")

        # Load trim config
        try:
            if self.trims_path:
                self.trim_config.load(self.trims_path)
            else:
                self.trim_config.load()
        except FileNotFoundError:
            print("WARNING: No trims config found. Use number keys or MQTT to set trim.")

        if self.trim_config.trims:
            print(f"Loaded {len(self.trim_config.trims)} trim config(s): "
                  f"{', '.join(self.trim_config.list_trims())}")

        # MQTT
        self.mqtt.set_trim_callback(self._on_trim_received)
        try:
            self.mqtt.connect()
        except Exception as e:
            print(f"MQTT connect failed ({e}), running offline.")
            self.mqtt = DummyMQTTClient()
            self.mqtt.set_trim_callback(self._on_trim_received)

        # Camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        prev_time = time.time()
        fps = 0.0

        print("\n=== RUN MODE ===")
        print("Press 1-9 to set trim, N to start cycle, ENTER to end cycle, Q to quit.\n")

        try:
            while True:
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    continue
                t_capture = time.perf_counter()

                # ── pose / wrist detection ──────────────────────
                poses = self.detector.detect_poses(frame)
                t_inference = time.perf_counter()

                # ── feed wrist into sequence tracker ────────────
                # Use the first detected person's best wrist
                step_event: Optional[StepEvent] = None
                if poses and self.tracker.is_active:
                    pose = poses[0]
                    # prefer right wrist, fall back to left
                    wrist = pose.right_wrist or pose.left_wrist
                    if wrist:
                        step_event = self.tracker.update(wrist.x, wrist.y)

                        # annotate which zone the wrist is in
                        zones_hit = self.zone_manager.get_zones_for_point(wrist.x, wrist.y)
                        if zones_hit:
                            zone_label = zones_hit[0]["name"]
                            cv2.putText(
                                frame, f"wrist -> {zone_label}",
                                (wrist.x + 15, wrist.y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
                            )
                t_zone = time.perf_counter()

                # ── handle step event ───────────────────────────
                if step_event:
                    if step_event.is_correct:
                        print(f"  Step {self.tracker.current_step_index}: "
                              f"picked {step_event.zone_name} [OK]")
                    else:
                        msg = (f"Expected {step_event.expected_zone}, "
                               f"got {step_event.zone_name}")
                        print(f"  !! DEVIATION: {msg}")
                        self.alert.trigger(msg)

                    if self.tracker.is_sequence_complete:
                        result = self.tracker.end_cycle()
                        if result:
                            print(f"\nCycle complete! Errors: {result.error_count}, "
                                  f"Time: {result.cycle_time:.1f}s")
                            self.mqtt.publish_cycle(result.to_dict())
                            if self._stats is not None:
                                self._stats.record_cycle(
                                    result.expected_sequence, result.observed_sequence,
                                    result.error_count, result.completed,
                                )

                # ── colour part detection per zone ──────────────
                zone_parts: Dict[str, List[PartDetection]] = defaultdict(list)
                for zone in self.zone_manager.zones:
                    mask = self.zone_manager.get_zone_mask(zone, frame.shape)
                    parts = self.detector.detect_colored_parts(frame, mask=mask)
                    zone_parts[zone["name"]].extend(parts)
                t_parts = time.perf_counter()

                # ── drawing ─────────────────────────────────────
                self.zone_manager.draw_zones(frame)
                self.detector.draw_poses(frame, poses)
                for parts_list in zone_parts.values():
                    self.detector.draw_parts(frame, parts_list)

                # trim banner (top of screen)
                self._draw_trim_banner(frame)

                # alert overlay (must be drawn last so it's on top)
                self.alert.draw(frame)

                # FPS
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now

                self._draw_status_panel(frame, fps, zone_parts)

                cv2.imshow(WINDOW_NAME, frame)
                t_render = time.perf_counter()

                # ── record stats ────────────────────────────────
                if self._stats is not None:
                    self._stats.record(
                        capture=(t_capture - t_start) * 1000,
                        inference=(t_inference - t_capture) * 1000,
                        zone_check=(t_zone - t_inference) * 1000,
                        part_detect=(t_parts - t_zone) * 1000,
                        render=(t_render - t_parts) * 1000,
                        total=(t_render - t_start) * 1000,
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                # number keys 1-9 → set trim
                elif ord("1") <= key <= ord("9"):
                    trim_names = self.trim_config.list_trims()
                    idx = key - ord("1")
                    if idx < len(trim_names):
                        self._current_trim = trim_names[idx]
                        print(f"Trim set to '{self._current_trim}'")
                    else:
                        print(f"No trim at index {idx + 1}")

                # 'n' → start new cycle with current trim
                elif key == ord("n"):
                    if not self._current_trim:
                        print("Set a trim first (press 1-9 or send via MQTT).")
                    else:
                        seq = self.trim_config.get_sequence(self._current_trim)
                        if seq:
                            self.tracker.start_cycle(self._current_trim, seq)
                            print(f"Cycle started — trim={self._current_trim}, "
                                  f"sequence={seq}")
                        else:
                            print(f"No sequence for trim '{self._current_trim}'")

                # Enter → end cycle
                elif key == 13:
                    result = self.tracker.end_cycle()
                    if result:
                        print(f"\nCycle ended manually. Errors: {result.error_count}, "
                              f"Time: {result.cycle_time:.1f}s")
                        self.mqtt.publish_cycle(result.to_dict())
                        if self._stats is not None:
                            self._stats.record_cycle(
                                result.expected_sequence, result.observed_sequence,
                                result.error_count, result.completed,
                            )
                    else:
                        print("No active cycle to end.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mqtt.disconnect()

            if self._stats is not None:
                self._stats.mqtt_payload_bytes = self.mqtt.bytes_sent_log
                self._stats.print_summary()
                self._stats.write_csv(self._stats_csv)
