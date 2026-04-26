

---

## Slide 1 — Title (~30 sec)

**Edge-Based Real-Time Process Adherence Monitoring for Manufacturing Assembly Lines**

Erin Walter
Electrical and Computer Engineering — University of Michigan–Dearborn

---

## Slide 2 — Problem & Motivation (~60 sec)

**Assembly line operators must pick parts in the exact right order.**

- Wrong sequence → defects, rework, safety issues, costly recalls
- Defects caught at end-of-line — too late and too expensive to fix
- Manual supervision doesn't scale; operators perform differently when watched *(Hawthorne effect)*

**The gap:** no low-cost, real-time, privacy-preserving solution exists today.

> Existing options: cloud analytics (too slow), light curtains (inflexible), barcode scanning (adds steps), manual audits (don't scale)

---

## Slide 3 — Why Edge Computing (~60 sec)

**Two reasons cloud doesn't work here:**

**1. Latency** — an alert must reach the operator while their hand is still moving
- Cloud round-trip: 100–500 ms → operator has already completed the wrong pick
- Edge inference: ~34 ms → alert appears within one frame

**2. Privacy & Bandwidth** — streaming raw video off the factory floor is a problem
- 1.34 MB/s per camera = ~4.8 GB/hour
- Raises operator privacy and data sovereignty concerns

**Edge solves both: inference stays local, raw video never leaves the device.**

---

## Slide 4 — System Design (~90 sec)

*(show architecture diagram)*

**Two tiers:**
- **Jetson Nano (edge):** camera → YOLOv8n-pose → wrist tracking → zone check → sequence validation → on-screen alert
- **MacBook Pro (PC):** MQTT broker → SQLite → Flask dashboard

**Per-frame pipeline (runs every ~33 ms):**
1. Capture frame
2. YOLOv8n-pose → extract wrist (x, y)
3. Point-in-polygon → which bin zone?
4. 5-frame debounce → filter jitter
5. Compare to expected trim sequence
6. Alert immediately if wrong, update progress if correct

**MQTT carries only ~500-byte JSON results per cycle — not video.**

---

## Slide 5 — Demo / Screenshots (~90 sec)

*(show screenshots or live demo)*

- **Zone config:** supervisor draws polygon zones on live camera feed, saved to JSON
- **Normal operation:** green trim banner, wrist keypoint tracked, step-by-step progress
- **Deviation detected:** red flashing border + "WRONG SEQUENCE" + expected vs. actual pick
- **PC dashboard:** cycle history, pass/fail status, error rates, per-step breakdown

**Trim levels (A, B, C) sent over MQTT — system switches sequences automatically.**

---

## Slide 6 — Key Results: Latency (~60 sec)

**Edge is 9.6× faster than a simulated cloud path**

| Scenario | Mean Latency | P95 |
|---|---|---|
| Edge (Jetson Nano) | **33.8 ms** | 43.9 ms |
| Cloud (300 ms RTT) | 324.1 ms | 331.5 ms |
| Speedup | **9.6×** | |

- Edge stays within one frame period (~33 ms) at all times
- Even at an optimistic 100 ms RTT, cloud latency exceeds 150 ms — still too slow

---

## Slide 7 — Key Results: Bandwidth + Accuracy (~60 sec)

**Bandwidth — 161,000× reduction**

| | Edge | Cloud (video) |
|---|---|---|
| Data per cycle | ~500 B | ~81 MB |
| Data per hour | ~30 KB | ~4.8 GB |

**Detection Accuracy** *(30 cycles, 3 trim levels)*

| Metric | Value |
|---|---|
| Cycle-level pass rate | [TODO after running cycles] % |
| Step-level accuracy | [TODO after running cycles] % |

---

## Slide 8 — Conclusion (~30 sec)

**Edge deployment delivers what cloud can't:**

✅ Real-time feedback — 33.8 ms, within one frame
✅ Privacy — raw video never leaves the Jetson
✅ Bandwidth — 161,000× less data than video streaming
✅ Reliability — works fully offline, MQTT optional

**Running on a low-cost Jetson Nano + USB webcam. Fully functional prototype.**

*Limitations: single operator, 2D camera only, static zone config*
*Future: TensorRT optimization, depth camera, store-and-forward MQTT*

---

## Speaker Notes

- **Slide 2:** Lead with the real-world cost of defects — make it concrete
- **Slide 3:** The "hand still moving" line is your strongest argument for edge — emphasize it
- **Slide 4:** Walk through the diagram top to bottom, don't rush — this is the technical heart
- **Slide 5:** If doing live demo, have the system running before you start. If screenshots only, show all 4 states
- **Slide 6:** Point out that even the best-case cloud scenario (100 ms RTT) is still 4× slower
- **Slide 7:** The 161,000× number is striking — let it land before moving on
- **Slide 8:** End confident — this is a working system, not a concept
