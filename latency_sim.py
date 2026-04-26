#!/usr/bin/env python3
"""
Latency simulation — edge vs cloud comparison without cv2 or ultralytics.

Edge latency: simulated using realistic Jetson Nano YOLOv8n-pose benchmarks
              (mean ~33ms @ 30fps with CUDA, std ~5ms).
Cloud latency: real MQTT round-trip to broker + simulated network RTT
               + simulated cloud inference time.

Outputs the Section 4.3 comparison table and per-sample CSV.

Usage:
    python latency_sim.py [--samples 100] [--mqtt-host localhost] [--network-rtt 150]
"""

import argparse
import csv
import json
import math
import random
import statistics
import threading
import time

import paho.mqtt.client as mqtt

TOPIC_REQ = "latsim/request"
TOPIC_RES = "latsim/response"

# Jetson Nano YOLOv8n-pose with CUDA — based on ultralytics published benchmarks
EDGE_INFERENCE_MEAN_MS  = 33.0
EDGE_INFERENCE_STD_MS   =  5.0
EDGE_ZONE_CHECK_MEAN_MS =  0.8
EDGE_ZONE_CHECK_STD_MS  =  0.2

# Cloud inference (server-class GPU, faster than Jetson)
CLOUD_INFERENCE_MEAN_MS = 20.0
CLOUD_INFERENCE_STD_MS  =  4.0


def _gauss_pos(mean: float, std: float) -> float:
    return max(1.0, random.gauss(mean, std))


# ── embedded echo server (runs in background thread) ────────────

class EchoServer:
    """Receives requests on TOPIC_REQ and immediately publishes back on TOPIC_RES."""

    def __init__(self, host: str, port: int, cloud_inference_ms: float) -> None:
        self._cloud_inference_ms = cloud_inference_ms
        self._client = mqtt.Client(client_id="latsim-server", protocol=mqtt.MQTTv311)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._host = host
        self._port = port

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_REQ)

    def _on_message(self, client, userdata, msg):
        # simulate cloud inference time
        infer_ms = _gauss_pos(CLOUD_INFERENCE_MEAN_MS, CLOUD_INFERENCE_STD_MS)
        time.sleep(infer_ms / 1000)
        # echo back with server timestamps
        try:
            payload = json.loads(msg.payload)
        except Exception:
            return
        payload["server_inference_ms"] = round(infer_ms, 2)
        payload["server_sent_at"] = time.perf_counter()
        client.publish(TOPIC_RES, json.dumps(payload))

    def start(self):
        self._client.connect(self._host, self._port, keepalive=60)
        self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()


# ── latency client ───────────────────────────────────────────────

class LatencyClient:
    def __init__(self, host: str, port: int, network_rtt_ms: float) -> None:
        self._network_rtt_ms = network_rtt_ms
        self._pending: dict[int, float] = {}
        self._results: dict[int, dict] = {}
        self._event = threading.Event()

        self._client = mqtt.Client(client_id="latsim-client", protocol=mqtt.MQTTv311)
        self._client.on_connect = lambda c, u, f, rc: c.subscribe(TOPIC_RES)
        self._client.on_message = self._on_message
        self._host = host
        self._port = port

    def _on_message(self, client, userdata, msg):
        recv_at = time.perf_counter()
        try:
            payload = json.loads(msg.payload)
        except Exception:
            return
        sid = payload.get("sample_id", -1)
        if sid in self._pending:
            mqtt_rtt_ms = (recv_at - self._pending[sid]) * 1000
            # add simulated one-way network latency (each direction = rtt/2)
            simulated_cloud_ms = mqtt_rtt_ms + self._network_rtt_ms
            self._results[sid] = {
                "mqtt_rtt_ms": mqtt_rtt_ms,
                "cloud_total_ms": simulated_cloud_ms,
                "server_inference_ms": payload.get("server_inference_ms", 0),
            }
            self._event.set()

    def connect(self):
        self._client.connect(self._host, self._port, keepalive=60)
        self._client.loop_start()

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def measure_cloud(self, sample_id: int, timeout: float = 10.0) -> float | None:
        self._event.clear()
        self._pending[sample_id] = time.perf_counter()
        payload = json.dumps({"sample_id": sample_id, "sent_at": self._pending[sample_id]})
        self._client.publish(TOPIC_REQ, payload)
        if self._event.wait(timeout=timeout):
            return self._results.get(sample_id, {}).get("cloud_total_ms")
        return None


# ── main ─────────────────────────────────────────────────────────

def run(num_samples: int, host: str, port: int, network_rtt_ms: float, output_csv: str, txt_output: str = "results_latency_edge.txt") -> None:
    print(f"Latency simulation — {num_samples} samples")
    print(f"  Edge model:    Jetson Nano YOLOv8n-pose (CUDA), mean={EDGE_INFERENCE_MEAN_MS}ms")
    print(f"  Cloud network: simulated RTT +{network_rtt_ms}ms  |  broker: {host}:{port}")
    print()

    # --- edge pass (pure simulation, no YOLO needed) ---
    print(f"=== EDGE PASS ({num_samples} samples) ===")
    edge_latencies: list[float] = []
    for _ in range(num_samples):
        infer  = _gauss_pos(EDGE_INFERENCE_MEAN_MS, EDGE_INFERENCE_STD_MS)
        zone   = _gauss_pos(EDGE_ZONE_CHECK_MEAN_MS, EDGE_ZONE_CHECK_STD_MS)
        edge_latencies.append(infer + zone)
    print(f"  Mean: {statistics.mean(edge_latencies):.1f}ms  "
          f"Median: {statistics.median(edge_latencies):.1f}ms\n")

    # --- cloud pass (real MQTT + simulated network) ---
    print(f"=== CLOUD PASS ({num_samples} samples) ===")
    server = EchoServer(host, port, CLOUD_INFERENCE_MEAN_MS)
    client = LatencyClient(host, port, network_rtt_ms)

    try:
        server.start()
        client.connect()
        time.sleep(0.3)  # let subscriptions propagate

        cloud_latencies: list[float] = []
        for i in range(num_samples):
            result = client.measure_cloud(i)
            if result is not None:
                cloud_latencies.append(result)
            else:
                print(f"  WARNING: sample {i} timed out")
                cloud_latencies.append(float("nan"))

        valid_cloud = [x for x in cloud_latencies if not math.isnan(x)]
        if valid_cloud:
            print(f"  Mean: {statistics.mean(valid_cloud):.1f}ms  "
                  f"Median: {statistics.median(valid_cloud):.1f}ms\n")
    finally:
        client.disconnect()
        server.stop()

    # --- results table ---
    valid_cloud = [x for x in cloud_latencies if not math.isnan(x)]

    def pct(data: list[float], p: float) -> float:
        s = sorted(data)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    print("=" * 62)
    print("LATENCY COMPARISON  (paper Section 4.3)")
    print("=" * 62)
    print(f"  {'Metric':<32} {'Edge (ms)':>12} {'Cloud (ms)':>12}")
    print("  " + "-" * 58)
    print(f"  {'Mean latency':<32} {statistics.mean(edge_latencies):>12.1f} {statistics.mean(valid_cloud):>12.1f}")
    print(f"  {'Median latency':<32} {statistics.median(edge_latencies):>12.1f} {statistics.median(valid_cloud):>12.1f}")
    print(f"  {'P95 latency':<32} {pct(edge_latencies, 95):>12.1f} {pct(valid_cloud, 95):>12.1f}")
    print(f"  {'Min latency':<32} {min(edge_latencies):>12.1f} {min(valid_cloud):>12.1f}")
    print(f"  {'Max latency':<32} {max(edge_latencies):>12.1f} {max(valid_cloud):>12.1f}")
    speedup = statistics.mean(valid_cloud) / statistics.mean(edge_latencies)
    print(f"\n  Speedup (cloud / edge): {speedup:.1f}x")
    print("=" * 62)

    print(f"""
Paper table values (copy-paste ready):
  Edge feedback latency   : ~{statistics.mean(edge_latencies):.0f} ms  (Jetson Nano, simulated)
  Cloud feedback latency  : ~{statistics.mean(valid_cloud):.0f} ms  (MQTT RTT + {network_rtt_ms}ms network + inference)
  Speedup                 : {speedup:.1f}x
  Note: edge numbers based on YOLOv8n-pose Jetson Nano benchmarks;
        run 'python -m src.main benchmark' on real Jetson for measured values.
""")

    # --- CSV ---
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "edge_ms", "cloud_ms"])
        for i in range(max(len(edge_latencies), len(cloud_latencies))):
            e = edge_latencies[i] if i < len(edge_latencies) else ""
            c = cloud_latencies[i] if i < len(cloud_latencies) else ""
            w.writerow([i, round(e, 2) if e != "" else "", round(c, 2) if c != "" else ""])
    print(f"Per-sample data written to {output_csv}")

    # --- text results ---
    lines = [
        "LATENCY SIMULATION RESULTS",
        f"Samples: {num_samples}  |  Network RTT: +{network_rtt_ms}ms  |  Broker: {host}:{port}",
        f"Edge model: Jetson Nano YOLOv8n-pose (CUDA), mean={EDGE_INFERENCE_MEAN_MS}ms",
        "",
        "=" * 62,
        "LATENCY COMPARISON  (paper Section 4.3)",
        "=" * 62,
        f"  {'Metric':<32} {'Edge (ms)':>12} {'Cloud (ms)':>12}",
        "  " + "-" * 58,
        f"  {'Mean latency':<32} {statistics.mean(edge_latencies):>12.1f} {statistics.mean(valid_cloud):>12.1f}",
        f"  {'Median latency':<32} {statistics.median(edge_latencies):>12.1f} {statistics.median(valid_cloud):>12.1f}",
        f"  {'P95 latency':<32} {pct(edge_latencies, 95):>12.1f} {pct(valid_cloud, 95):>12.1f}",
        f"  {'Min latency':<32} {min(edge_latencies):>12.1f} {min(valid_cloud):>12.1f}",
        f"  {'Max latency':<32} {max(edge_latencies):>12.1f} {max(valid_cloud):>12.1f}",
        f"\n  Speedup (cloud / edge): {speedup:.1f}x",
        "=" * 62,
        "",
        "Paper table values:",
        f"  Edge feedback latency   : ~{statistics.mean(edge_latencies):.0f} ms  (Jetson Nano, simulated)",
        f"  Cloud feedback latency  : ~{statistics.mean(valid_cloud):.0f} ms  (MQTT RTT + {network_rtt_ms}ms network + inference)",
        f"  Speedup                 : {speedup:.1f}x",
        "  Note: edge numbers based on YOLOv8n-pose Jetson Nano benchmarks;",
        "        run 'python -m src.main benchmark' on real Jetson for measured values.",
    ]
    with open(txt_output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Text results written to {txt_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency simulation (no cv2/ultralytics needed)")
    parser.add_argument("--samples",     type=int,   default=100,
                        help="Number of samples (default: 100)")
    parser.add_argument("--mqtt-host",   type=str,   default="localhost")
    parser.add_argument("--mqtt-port",   type=int,   default=1883)
    parser.add_argument("--network-rtt", type=float, default=150.0,
                        help="Simulated one-way cloud network latency in ms (default: 150)")
    parser.add_argument("--output",      type=str,   default="latency_sim_results.csv")
    parser.add_argument("--txt-output",  type=str,   default="results_latency_edge.txt")
    args = parser.parse_args()

    run(args.samples, args.mqtt_host, args.mqtt_port, args.network_rtt, args.output, args.txt_output)


if __name__ == "__main__":
    main()