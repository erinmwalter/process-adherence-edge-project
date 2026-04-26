#!/usr/bin/env python3
"""
Cloud simulation server — receives frames via MQTT, runs the same
YOLO + zone-check pipeline, and publishes the result back.

This simulates a cloud-based video analytics service to benchmark
edge vs cloud latency.

Usage:
  python -m src.benchmark_server --mqtt-host localhost [--config config/zones.json]
"""

import argparse
import base64
import json
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from src.detector import Detector
from src.zone_manager import ZoneManager
from src.sequence_tracker import SequenceTracker

TOPIC_FRAME = "benchmark/frame"
TOPIC_RESULT = "benchmark/result"


class CloudSimServer:
    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        config_path: str | None,
        yolo_model: str,
        confidence: float,
    ) -> None:
        self.zone_manager = ZoneManager()
        self.detector = Detector(yolo_model=yolo_model, confidence=confidence)
        self.tracker = SequenceTracker(self.zone_manager)
        self.config_path = config_path

        self._client = mqtt.Client(protocol=mqtt.MQTTv311)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._broker_host = broker_host
        self._broker_port = broker_port

        self._frames_processed = 0

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_FRAME)
            print(f"[Cloud Server] Connected to broker, subscribed to {TOPIC_FRAME}")
        else:
            print(f"[Cloud Server] Connection failed, rc={rc}")

    def _on_message(self, client, userdata, msg):
        """Receive a frame, run inference, publish result."""
        recv_time = time.perf_counter()

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        frame_id = payload.get("frame_id", -1)
        sent_at = payload.get("sent_at", 0.0)

        # decode JPEG frame
        jpg_bytes = base64.b64decode(payload["frame_b64"])
        frame = cv2.imdecode(
            np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            return

        # --- run the same pipeline the edge runs ---
        t0 = time.perf_counter()

        poses = self.detector.detect_poses(frame)

        zone_name = None
        if poses:
            pose = poses[0]
            wrist = pose.right_wrist or pose.left_wrist
            if wrist:
                zones_hit = self.zone_manager.get_zones_for_point(wrist.x, wrist.y)
                pick_zones = [z for z in zones_hit if z["zone_type"].startswith("pick")]
                if pick_zones:
                    zone_name = pick_zones[0]["name"]

        inference_time = time.perf_counter() - t0

        # publish result back
        result = {
            "frame_id": frame_id,
            "sent_at": sent_at,
            "server_recv_at": recv_time,
            "inference_time_ms": round(inference_time * 1000, 2),
            "zone": zone_name,
            "server_sent_at": time.perf_counter(),
        }
        self._client.publish(TOPIC_RESULT, json.dumps(result))

        self._frames_processed += 1
        if self._frames_processed % 10 == 0:
            print(f"[Cloud Server] Processed {self._frames_processed} frames "
                  f"(last inference: {inference_time*1000:.1f}ms)")

    def run(self):
        # load zones
        try:
            if self.config_path:
                self.zone_manager.load(self.config_path)
            else:
                self.zone_manager.load()
            print(f"[Cloud Server] Loaded {len(self.zone_manager.zones)} zone(s)")
        except FileNotFoundError:
            print("[Cloud Server] WARNING: No zone config found, zone checks will be empty")

        self._client.connect(self._broker_host, self._broker_port, keepalive=60)
        print(f"[Cloud Server] Listening on {self._broker_host}:{self._broker_port}...")
        print("[Cloud Server] Press Ctrl+C to stop.\n")
        self._client.loop_forever()


def main():
    parser = argparse.ArgumentParser(description="Cloud simulation server for benchmarking")
    parser.add_argument("--mqtt-host", type=str, default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to zones.json (default: config/zones.json)")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.45)
    args = parser.parse_args()

    server = CloudSimServer(
        broker_host=args.mqtt_host,
        broker_port=args.mqtt_port,
        config_path=args.config,
        yolo_model=args.model,
        confidence=args.conf,
    )
    server.run()


if __name__ == "__main__":
    main()
