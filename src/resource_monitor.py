#!/usr/bin/env python3
"""
Resource monitor for NVIDIA Jetson Nano.

Logs CPU, GPU, memory, and temperature from tegrastats at a configurable
interval.  Produces a CSV suitable for filling Table V (Resource Utilization)
in the paper.

Usage (run alongside the edge application):
    python -m src.resource_monitor --duration 60 --interval 1 --output resource_stats.csv

Requires: tegrastats (pre-installed on JetPack).
Falls back to /proc-based collection on non-Jetson Linux for testing.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Sample:
    timestamp: float
    cpu_percent: float          # average across cores
    mem_used_mb: float
    mem_total_mb: float
    gpu_percent: float
    temp_cpu_c: Optional[float] = None
    temp_gpu_c: Optional[float] = None


# ── tegrastats parser ────────────────────────────────────────────

_RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_RE_CPU = re.compile(r"\[([^\]]+)\]")          # e.g. [25%@1479,30%@1479,...]
_RE_GPU = re.compile(r"GR3D_FREQ (\d+)%")
_RE_TEMP_CPU = re.compile(r"CPU@([\d.]+)C")
_RE_TEMP_GPU = re.compile(r"GPU@([\d.]+)C")


def parse_tegrastats_line(line: str) -> Optional[Sample]:
    """Parse a single tegrastats output line into a Sample."""
    ram = _RE_RAM.search(line)
    cpu_match = _RE_CPU.search(line)
    gpu = _RE_GPU.search(line)

    if not (ram and cpu_match):
        return None

    mem_used = int(ram.group(1))
    mem_total = int(ram.group(2))

    # parse per-core cpu percentages
    core_strs = cpu_match.group(1).split(",")
    percents = []
    for cs in core_strs:
        m = re.match(r"(\d+)%", cs.strip())
        if m:
            percents.append(int(m.group(1)))
    cpu_avg = sum(percents) / len(percents) if percents else 0.0

    gpu_pct = int(gpu.group(1)) if gpu else 0.0

    temp_cpu_m = _RE_TEMP_CPU.search(line)
    temp_gpu_m = _RE_TEMP_GPU.search(line)

    return Sample(
        timestamp=time.time(),
        cpu_percent=cpu_avg,
        mem_used_mb=mem_used,
        mem_total_mb=mem_total,
        gpu_percent=gpu_pct,
        temp_cpu_c=float(temp_cpu_m.group(1)) if temp_cpu_m else None,
        temp_gpu_c=float(temp_gpu_m.group(1)) if temp_gpu_m else None,
    )


# ── /proc fallback for non-Jetson Linux / macOS testing ─────────

def _read_proc_sample() -> Sample:
    """Best-effort resource sample using /proc (Linux) or psutil-free fallback."""
    import os

    cpu_pct = 0.0
    mem_used = 0.0
    mem_total = 0.0
    gpu_pct = 0.0

    # CPU — quick snapshot via /proc/stat (Linux)
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        idle = int(parts[4])
        total = sum(int(p) for p in parts[1:])
        cpu_pct = (1 - idle / total) * 100 if total else 0.0
    except Exception:
        pass

    # Memory
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for ln in lines:
            k, v = ln.split(":")[:2]
            info[k.strip()] = int(v.strip().split()[0])
        mem_total = info.get("MemTotal", 0) / 1024
        mem_used = mem_total - info.get("MemAvailable", 0) / 1024
    except Exception:
        pass

    return Sample(
        timestamp=time.time(),
        cpu_percent=round(cpu_pct, 1),
        mem_used_mb=round(mem_used, 1),
        mem_total_mb=round(mem_total, 1),
        gpu_percent=gpu_pct,
    )


# ── collection loop ─────────────────────────────────────────────

def collect_tegrastats(duration_s: int, interval_ms: int) -> List[Sample]:
    """Run tegrastats for *duration_s* seconds, return samples."""
    samples: List[Sample] = []
    cmd = ["tegrastats", "--interval", str(interval_ms)]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except FileNotFoundError:
        print("tegrastats not found — using /proc fallback.", file=sys.stderr)
        end = time.time() + duration_s
        while time.time() < end:
            samples.append(_read_proc_sample())
            time.sleep(interval_ms / 1000)
        return samples

    end = time.time() + duration_s
    try:
        while time.time() < end:
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                break
            s = parse_tegrastats_line(line)
            if s:
                samples.append(s)
    finally:
        proc.terminate()
        proc.wait()

    return samples


# ── summary + CSV ────────────────────────────────────────────────

def print_summary(samples: List[Sample]) -> None:
    import statistics

    if not samples:
        print("No samples collected.")
        return

    cpus = [s.cpu_percent for s in samples]
    mems = [s.mem_used_mb for s in samples]
    gpus = [s.gpu_percent for s in samples]
    mem_total = samples[0].mem_total_mb

    print(f"\n{'=' * 56}")
    print(f"RESOURCE UTILIZATION ({len(samples)} samples)")
    print(f"{'=' * 56}")
    print(f"{'Metric':<28} {'Mean':>8} {'Max':>8}")
    print(f"{'-' * 56}")
    print(f"{'CPU utilization':<28} {statistics.mean(cpus):>7.1f}% {max(cpus):>7.1f}%")
    print(f"{'Memory usage (MB)':<28} {statistics.mean(mems):>7.0f}   {max(mems):>7.0f}")
    print(f"{'Memory total (MB)':<28} {mem_total:>7.0f}")
    print(f"{'GPU utilization':<28} {statistics.mean(gpus):>7.1f}% {max(gpus):>7.1f}%")

    temps = [s.temp_cpu_c for s in samples if s.temp_cpu_c is not None]
    if temps:
        print(f"{'CPU temp (°C)':<28} {statistics.mean(temps):>7.1f}  {max(temps):>7.1f}")
    temps_gpu = [s.temp_gpu_c for s in samples if s.temp_gpu_c is not None]
    if temps_gpu:
        print(f"{'GPU temp (°C)':<28} {statistics.mean(temps_gpu):>7.1f}  {max(temps_gpu):>7.1f}")
    print(f"{'=' * 56}")


def write_csv(samples: List[Sample], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "cpu_percent", "mem_used_mb", "mem_total_mb",
            "gpu_percent", "temp_cpu_c", "temp_gpu_c",
        ])
        for s in samples:
            w.writerow([
                round(s.timestamp, 3),
                round(s.cpu_percent, 1),
                round(s.mem_used_mb, 1),
                round(s.mem_total_mb, 1),
                round(s.gpu_percent, 1),
                round(s.temp_cpu_c, 1) if s.temp_cpu_c is not None else "",
                round(s.temp_gpu_c, 1) if s.temp_gpu_c is not None else "",
            ])
    print(f"Resource samples written to {path}")


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log Jetson Nano resource utilization during edge inference."
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Monitoring duration in seconds (default: 60)")
    parser.add_argument("--interval", type=int, default=1000,
                        help="Sample interval in milliseconds (default: 1000)")
    parser.add_argument("--output", type=str, default="resource_stats.csv",
                        help="Output CSV path (default: resource_stats.csv)")
    args = parser.parse_args()

    print(f"Monitoring resources for {args.duration}s (interval {args.interval}ms)...")
    samples = collect_tegrastats(args.duration, args.interval)
    print_summary(samples)
    write_csv(samples, args.output)


if __name__ == "__main__":
    main()
