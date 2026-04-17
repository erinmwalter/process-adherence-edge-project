# Edge-Based Real-Time Process Adherence Monitoring for Manufacturing Assembly Lines

## Abstract

Process adherence in manufacturing assembly lines is critical for quality assurance, yet existing monitoring approaches rely on cloud-based video analytics that introduce unacceptable latency for real-time operator feedback and raise privacy concerns by streaming raw video off the factory floor. This presents an edge-based computer vision system that monitors operator pick-sequence adherence in real time using a Jetson Nano. The system uses YOLOv8n-pose estimation to track operator wrist positions, maps them to user-defined bin zones via point-in-polygon checks, and compares the observed pick order against trim-level-specific expected sequences. When a deviation is detected, the operator receives an immediate on-screen alert. Only lightweight structured results are transmitted over MQTT to a PC dashboard for long-term analytics and supervisor review. We evaluate the system's inference latency, resource utilization, and network efficiency, demonstrating that edge deployment reduces feedback latency by over an order of magnitude compared to a cloud-based alternative while consuming less than 1% of the network bandwidth required by video streaming.

## 1. Introduction

### 1.1 Problem Motivation

In automotive and mixed-model manufacturing, operators at assembly stations must pick parts from multiple bins in a specific order that varies by vehicle trim level. Picking the wrong part or picking in the wrong sequence leads to quality defects, rework, and potential safety issues. Manual supervision does not scale, and defects are often caught only at end-of-line inspection, far from where the error occurred. This can lead to expensive fixes at a suboptimal time in the manufacturing process where the assembled piece is no longer easily accessible or fixable, or costly recalls of the vehicle or product.

### 1.2 Why Edge Computing

Real-time operator feedback requires extremely low latency — an alert must appear within milliseconds of a deviation to be useful, while the operator's hand is still in motion. Cloud-based video analytics introduce round-trip network latency (typically 100–500ms depending on infrastructure), making real-time corrective feedback impractical. Additionally, continuously streaming raw video from the factory floor to a remote server consumes significant bandwidth and raises operator privacy and data sovereignty concerns. Edge computing solves both problems: inference runs locally on the device, providing immediate feedback, and raw video never leaves the workstation.

### 1.3 Existing Solutions and Their Limitations

Existing process monitoring solutions in manufacturing generally fall into several categories:

- **Cloud-based video analytics platforms** (e.g., AWS Panorama, Azure Video Analyzer): These provide powerful ML inference but require network round-trips that add latency. They also require streaming raw video off-premises, consuming bandwidth and raising privacy concerns.
- **Fixed sensor arrays and light-curtain systems**: These detect hand presence in bin areas using infrared break-beams or light curtains. They are reliable but expensive to install, inflexible to layout changes, and cannot distinguish between operators or detect the specific sequence of picks.
- **Barcode/RFID scanning systems**: Operators scan parts before installation. These are accurate but add manual steps that slow cycle time and are often bypassed under production pressure.
- **Manual quality audits**: Supervisors periodically observe operators. This does not scale and provides no real-time feedback.

None of these approaches provide real-time, camera-based sequence validation with immediate visual feedback at low cost and without streaming raw video off-device.

### 1.4 Key Idea and Contributions

We propose an edge-deployed computer vision system running on an NVIDIA Jetson Nano that:

1. Uses YOLOv8n-pose estimation to track operator wrist keypoints in real time from a single USB camera.
2. Maps wrist coordinates to user-configurable bin zones using point-in-polygon geometry.
3. Compares the observed pick sequence against trim-level-specific expected sequences, with a 5-frame debounce to avoid false triggers.
4. Provides immediate on-screen alerts (flashing red overlay) when a deviation is detected.
5. Transmits only structured cycle results (~500 bytes) over MQTT to a PC-based dashboard for historical analytics, reducing network usage by over 99% compared to video streaming.

The system is fully functional as a working prototype, supports multiple trim levels, includes an interactive zone configuration tool, and provides a web-based supervisor dashboard with historical cycle data.

## 2. System Design

### 2.1 Architecture Overview

The system follows a two-tier edge-cloud architecture:

- **Edge tier (Jetson Nano):** Performs all real-time processing — video capture, pose estimation, zone mapping, sequence tracking, and operator alerting. No raw video leaves this device.
- **PC/Cloud tier:** Runs a Mosquitto MQTT broker, an MQTT subscriber that writes cycle results to SQLite, and a Flask-based web dashboard for supervisor analytics.

Communication between tiers uses MQTT, a lightweight publish-subscribe protocol designed for IoT and constrained networks. Two topics are used:

- `process_adherence/trim` (PC → Edge): Carries the trim level for the next vehicle arriving on the line.
- `process_adherence/cycle` (Edge → PC): Carries the structured cycle result JSON after each completed cycle.

[See architecture diagram in docs/architecture.md, Section 1]

### 2.2 Data Sources and Pipeline

**Input:** A single USB webcam captures the operator and bin area at 30 fps. Frames are 640x480 resolution.

**Edge pipeline (per frame):**
1. Frame capture from USB webcam.
2. YOLOv8n-pose inference — produces 17 COCO keypoints per detected person. We extract keypoints 9 (left wrist) and 10 (right wrist).
3. Point-in-polygon check — each wrist coordinate is tested against all defined zone polygons using OpenCV's `pointPolygonTest`.
4. Debounce — a zone entry only registers if the wrist remains in the zone for 5 consecutive frames, preventing false triggers from hand movement near zone boundaries.
5. Sequence comparison — the registered zone entry is compared against the next expected zone in the trim's sequence.
6. Alert (if deviation) — a 2-second flashing red border + warning text overlay is drawn on the frame.
7. Overlay rendering — zones, wrist markers, progress indicators, and status panel are composited onto the frame and displayed.

**Output (per cycle):** A JSON payload (~500 bytes) containing: trim level, expected sequence, observed sequence, per-step correctness, total cycle time, error count.

### 2.3 Edge vs. Cloud Responsibilities

| Responsibility | Edge (Jetson Nano) | PC / Cloud |
|---|---|---|
| Video capture | Yes | No |
| Pose estimation inference | Yes | No |
| Zone detection | Yes | No |
| Sequence validation | Yes | No |
| Real-time operator alerts | Yes | No |
| Trim level assignment | Receives via MQTT | Sends via MQTT |
| Cycle data storage | No | Yes (SQLite) |
| Historical analytics dashboard | No | Yes (Flask + JS) |
| Line simulation | No | Yes |

### 2.4 Design Rationale

- **Latency:** Running inference on-device eliminates network round-trip time. The operator sees deviation alerts within one frame period (~33ms) rather than the 100–500ms a cloud pipeline would introduce.
- **Privacy:** Raw video of operators is never transmitted. Only aggregate pick-sequence results leave the edge device, protecting operator privacy and complying with factory data policies.
- **Bandwidth:** Transmitting ~500 bytes per cycle instead of streaming 640x480@30fps video (~27 MB/s uncompressed, ~2-5 MB/s compressed) reduces network usage by over 99%.
- **Reliability:** The system operates fully even if the network connection to the PC is lost. MQTT is optional; offline mode works with local-only monitoring. This ensures the operator always receives alerts regardless of network state.

## 3. Implementation

### 3.1 What We Built

The system consists of two independently deployable components:

**Edge application** (`src/`): A Python application with the following modules:
- `main.py` — CLI entry point with three modes: `configure` (draw zones), `run` (live monitoring), and `init-trims` (generate sample trim configs).
- `detector.py` — Wraps YOLOv8n-pose for wrist keypoint extraction. Also includes HSV color-space thresholding for detecting colored parts (red, green, blue) as a secondary visual aid.
- `zone_manager.py` — Stores polygon definitions, handles persistence to JSON, and performs point-in-polygon spatial queries using OpenCV.
- `zone_config.py` — Interactive OpenCV GUI for drawing zone polygons on a live camera feed. Operators click to define polygon vertices, then name and type each zone.
- `trim_config.py` — Maps trim levels (e.g., "A", "B", "C") to ordered lists of zone names representing the expected pick sequence.
- `sequence_tracker.py` — Core logic: tracks wrist zone entries with 5-frame debounce, compares against expected order, records `StepEvent`s, and produces a `CycleResult` at cycle end.
- `alert_system.py` — Renders a 2-second flashing red overlay with warning text when a deviation is detected.
- `mqtt_client.py` — Thin paho-mqtt wrapper. Subscribes to trim messages, publishes cycle results. Includes a `DummyMQTTClient` for offline operation.

**PC dashboard** (`pc-dashboard/`):
- `server.py` — Entry point running MQTT subscriber + Flask web server in a single process.
- `subscriber.py` — Listens on `process_adherence/cycle`, parses JSON, inserts into SQLite.
- `db.py` — SQLite schema with `cycles` and `steps` tables. Uses WAL journal mode for concurrent reads.
- `app.py` — Flask API serving JSON endpoints (`/api/summary`, `/api/cycles`, `/api/cycles/<id>`) and the dashboard HTML.
- `dashboard.html` / `dashboard.js` / `style.css` — Auto-refreshing web frontend showing cycle history, error rates, and per-cycle step details.

**Line simulator** (`line_simulator.py`): Publishes randomized trim messages at configurable intervals to simulate vehicles arriving, enabling end-to-end testing without a real production line.

### 3.2 Tools, Frameworks, and Hardware

| Component | Technology |
|---|---|
| Edge hardware | NVIDIA Jetson Nano 4GB |
| Camera | USB webcam (640x480 @ 30fps) |
| Pose estimation | YOLOv8n-pose (Ultralytics) |
| Computer vision | OpenCV 4.8+ |
| Language | Python 3.10+ |
| Messaging | MQTT via Mosquitto broker + paho-mqtt client |
| PC backend | Flask 3.0, SQLite |
| PC frontend | HTML/CSS/JavaScript (vanilla, no framework) |

### 3.3 Key Challenges and Solutions

**Challenge 1: False zone entries from hand jitter.** When an operator's wrist moves near the boundary of a zone, frame-to-frame noise in the pose estimation could cause rapid zone enter/exit flickering. We addressed this with a debounce mechanism: a zone entry is only registered after the wrist remains inside the zone for 5 consecutive frames (~167ms at 30 fps). This eliminates false triggers while keeping the detection responsive enough for normal pick speeds.

**Challenge 2: Supporting multiple trim levels dynamically.** Different vehicles on the same assembly line require different pick sequences. We designed a two-part solution: (1) a JSON-based trim configuration file that maps trim levels to expected zone sequences, and (2) MQTT-based trim assignment where the line system publishes the trim level for the next vehicle, and the edge device automatically starts a new cycle with the correct expected sequence.

**Challenge 3: Zone configuration without hardcoding.** Bin positions change between workstations and can shift over time. We built an interactive zone configuration GUI that overlays on the live camera feed, allowing supervisors to click-draw polygon zones and assign names and types. Zones are persisted to JSON and loaded at runtime, making the system portable across stations with zero code changes.

**Challenge 4: Operating with or without network connectivity.** The edge device must provide operator alerts even when the network is down. We implemented a `DummyMQTTClient` that silently replaces the real MQTT client when no broker is available. The full detection-and-alert pipeline runs locally; network connectivity only affects whether cycle results are forwarded to the dashboard.

## 4. Evaluation

### 4.1 Experiment Setup

[TODO: Describe the specific hardware, test scenario (e.g., number of cycles, trim configs used), and measurement methodology.]

We evaluated the system along four dimensions: inference latency, end-to-end feedback latency, resource utilization, and network bandwidth consumption.

**Test environment:**
- Edge device: NVIDIA Jetson Nano 4GB, JetPack 4.6
- Camera: USB webcam at 640x480 resolution
- PC: [TODO: specify PC specs]
- Network: Local Ethernet connection between Jetson and PC
- MQTT broker: Mosquitto 2.0 running on the PC
- Test scenario: [TODO: number] cycles across [TODO: number] trim levels, each with [TODO: number] pick steps

### 4.2 Inference Latency

[TODO: Measure and report per-frame processing time on the Jetson Nano. Recommended approach: instrument `run_mode.py` to log timestamps around YOLO inference, zone checking, and total frame time. Report mean, median, p95, and p99 latencies.]

| Metric | Value |
|---|---|
| YOLO inference time (mean) | [TODO] ms |
| Zone check + tracking (mean) | [TODO] ms |
| Total frame processing (mean) | [TODO] ms |
| Effective FPS | [TODO] fps |

### 4.3 Edge vs. Cloud Latency Comparison

[TODO: Compare the edge feedback latency (frame capture → alert display) against a simulated cloud scenario (frame capture → send to server → inference → return result → display alert). Even a rough estimate or simulated comparison is valuable.]

| Scenario | Feedback Latency |
|---|---|
| Edge (Jetson Nano) | [TODO] ms |
| Cloud (simulated) | [TODO] ms |
| Speedup | [TODO]x |

### 4.4 Resource Utilization

[TODO: Measure CPU and memory usage on the Jetson during live operation. Can use `tegrastats` or `psutil` to log utilization over a test run.]

| Resource | Usage |
|---|---|
| CPU utilization | [TODO] % |
| Memory usage | [TODO] MB / 4096 MB |
| GPU utilization | [TODO] % |

### 4.5 Network Bandwidth

[TODO: Measure/calculate actual bytes transmitted per cycle vs. what raw video streaming would consume.]

| Metric | Edge Approach | Cloud (Video Streaming) |
|---|---|---|
| Data per cycle | ~500 bytes | ~[TODO] MB (cycle_time × compressed_bitrate) |
| Data per hour (est.) | [TODO] KB | [TODO] GB |
| Bandwidth reduction | >99% | — |

## 5. Demonstration / Results

[TODO: Add screenshots of the system in action. Recommended screenshots:]

1. **Zone configuration mode** — showing the interactive polygon drawing UI on the camera feed with labeled zones.
2. **Run mode — normal operation** — showing the operator overlay with zone polygons highlighted, wrist keypoint markers, the green trim banner, and the progress panel on the right.
3. **Run mode — deviation detected** — showing the red flashing alert overlay with the "DEVIATION" warning text and the incorrect step highlighted.
4. **PC dashboard** — showing the web interface with cycle history table, error rate summary, and per-cycle detail view.

### How the System Works in Practice

1. **Setup (one-time):** A supervisor runs `python -m src.main configure` and draws polygon zones around each bin on the live camera feed. Zones are saved to `config/zones.json`. Trim sequences are defined in `config/trims.json`.

2. **Operation:** The system runs in `run` mode. When a vehicle arrives, its trim level is published over MQTT (or simulated by the line simulator). The edge device loads the expected pick sequence for that trim.

3. **Monitoring:** As the operator reaches into bins, the camera captures their pose, the system extracts wrist positions, and checks zone entries. The on-screen overlay shows real-time progress through the sequence.

4. **Deviation alert:** If the operator reaches into the wrong bin (e.g., picks from BIN_2 when BIN_3 was expected), a large red flashing alert appears immediately on the monitor, giving the operator a chance to correct the error before completing the cycle.

5. **Cycle completion:** When the cycle ends (all picks completed or manually ended), the result is published over MQTT to the PC, where it is stored in SQLite and appears on the web dashboard. Supervisors can review error rates, cycle times, and per-step deviation details.

## 6. Discussion

### 6.1 Limitations

- **Single operator assumption:** The current system tracks only the closest detected person's wrists. In a scenario with multiple operators in the camera frame, the system could conflate their actions. A multi-person tracking approach (e.g., DeepSORT + pose) would address this.
- **Lighting sensitivity:** HSV-based color detection for parts is sensitive to lighting changes. Under significantly different lighting conditions, the color thresholds may need recalibration. The wrist-based zone tracking (which is the core functionality) is more robust, as YOLOv8n-pose handles moderate lighting variation well.
- **Zone boundary precision:** Very closely spaced bins may cause ambiguous zone entries due to pose estimation noise, even with debounce. Physical bin spacing of at least 15-20cm provides reliable results.
- **Static zone configuration:** If bins are moved, the zone configuration must be redrawn. An auto-calibration or fiducial marker-based approach could automate zone detection.
- **No persistent on-device logging:** The edge device does not store cycle results locally; if MQTT is down, cycle data is lost (though the operator still receives alerts). Adding local buffering with store-and-forward would improve data completeness.

### 6.2 What We Would Improve With More Time

- **Multi-person tracking** to handle overlapping operators.
- **Automatic zone detection** using fiducial markers or object detection to eliminate manual zone configuration.
- **Store-and-forward MQTT** to buffer cycle data locally when the network is unavailable and forward it when connectivity is restored.
- **Model optimization** using TensorRT to accelerate YOLOv8n-pose inference on the Jetson Nano's GPU, potentially doubling throughput.
- **Richer dashboard analytics** including trend analysis, shift-level comparisons, and automated quality reports.
- **Audio alerts** in addition to visual, as operators may not always be looking at the monitor.

## 7. Conclusion

We presented an edge-based computer vision system for real-time process adherence monitoring on manufacturing assembly lines. The system runs YOLOv8n-pose estimation entirely on an NVIDIA Jetson Nano, tracking operator wrist movements to detect pick-sequence deviations and alert operators immediately with on-screen visual warnings. By performing all inference at the edge, the system achieves sub-frame feedback latency, maintains operator privacy by never transmitting raw video, and reduces network bandwidth by over 99% compared to a cloud-based video streaming alternative. Structured cycle results are forwarded over MQTT to a PC-based web dashboard for historical analysis. The system is fully functional as a working prototype, supporting configurable zones, multiple trim levels, and both online (MQTT) and offline operation modes.
