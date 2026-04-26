#!/usr/bin/env python3
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Adherence Edge System",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── configure sub-command ────────────────────────────────────
    cfg_parser = subparsers.add_parser(
        "configure", help="Open zone configuration UI to draw and label zones."
    )
    cfg_parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    cfg_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to zone config JSON (default: config/zones.json)",
    )

    # ── run sub-command ──────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Run live process-adherence monitoring."
    )
    run_parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    run_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to zone config JSON (default: config/zones.json)",
    )
    run_parser.add_argument(
        "--trims", type=str, default=None,
        help="Path to trim sequences JSON (default: config/trims.json)",
    )
    run_parser.add_argument(
        "--model", type=str, default="yolov8n-pose.pt",
        help="YOLO model path or name (default: yolov8n-pose.pt)",
    )
    run_parser.add_argument(
        "--conf", type=float, default=0.45,
        help="YOLO confidence threshold (default: 0.45)",
    )
    run_parser.add_argument(
        "--mqtt-host", type=str, default=None,
        help="MQTT broker hostname (omit for offline mode)",
    )
    run_parser.add_argument(
        "--mqtt-port", type=int, default=1883,
        help="MQTT broker port (default: 1883)",
    )
    run_parser.add_argument(
        "--stats", action="store_true",
        help="Collect per-frame pipeline latency stats and write CSV on exit.",
    )
    run_parser.add_argument(
        "--stats-csv", type=str, default="pipeline_stats.csv",
        help="Output path for stats CSV (default: pipeline_stats.csv)",
    )

    # ── init-trims sub-command ───────────────────────────────────
    trims_parser = subparsers.add_parser(
        "init-trims", help="Generate a sample trims.json config for testing."
    )
    trims_parser.add_argument(
        "--trims", type=str, default=None,
        help="Output path (default: config/trims.json)",
    )

    # ── simulate sub-command ─────────────────────────────────────
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Simulate edge vs cloud latency and bandwidth without a camera.",
    )
    sim_parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of synthetic frames to benchmark (default: 100)",
    )
    sim_parser.add_argument(
        "--mqtt-host", type=str, default=None,
        help="MQTT broker hostname for cloud pass (omit to skip cloud comparison)",
    )
    sim_parser.add_argument(
        "--mqtt-port", type=int, default=1883,
        help="MQTT broker port (default: 1883)",
    )
    sim_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to zone config JSON (default: config/zones.json)",
    )
    sim_parser.add_argument(
        "--model", type=str, default="yolov8n-pose.pt",
        help="YOLO model path or name (default: yolov8n-pose.pt)",
    )
    sim_parser.add_argument(
        "--conf", type=float, default=0.45,
        help="YOLO confidence threshold (default: 0.45)",
    )
    sim_parser.add_argument(
        "--output", type=str, default="benchmark_results.csv",
        help="Output CSV path (default: benchmark_results.csv)",
    )

    # ── benchmark sub-command ────────────────────────────────────
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark edge vs cloud inference latency."
    )
    bench_parser.add_argument("--camera", type=int, default=0)
    bench_parser.add_argument("--frames", type=int, default=100,
                              help="Number of frames to benchmark (default: 100)")
    bench_parser.add_argument("--mqtt-host", type=str, default="localhost")
    bench_parser.add_argument("--mqtt-port", type=int, default=1883)
    bench_parser.add_argument("--config", type=str, default=None)
    bench_parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    bench_parser.add_argument("--conf", type=float, default=0.45)
    bench_parser.add_argument("--output", type=str, default="benchmark_results.csv")

    # ── benchmark-server sub-command ─────────────────────────────
    bsrv_parser = subparsers.add_parser(
        "benchmark-server", help="Run the cloud simulation server for benchmarking."
    )
    bsrv_parser.add_argument("--mqtt-host", type=str, default="localhost")
    bsrv_parser.add_argument("--mqtt-port", type=int, default=1883)
    bsrv_parser.add_argument("--config", type=str, default=None)
    bsrv_parser.add_argument("--model", type=str, default="yolov8n-pose.pt")
    bsrv_parser.add_argument("--conf", type=float, default=0.45)

    # ── resource-monitor sub-command ─────────────────────────────
    res_parser = subparsers.add_parser(
        "resource-monitor",
        help="Log CPU/GPU/memory utilization on Jetson Nano.",
    )
    res_parser.add_argument("--duration", type=int, default=60,
                            help="Monitoring duration in seconds (default: 60)")
    res_parser.add_argument("--interval", type=int, default=1000,
                            help="Sample interval in milliseconds (default: 1000)")
    res_parser.add_argument("--output", type=str, default="resource_stats.csv",
                            help="Output CSV path (default: resource_stats.csv)")

    args = parser.parse_args()

    if args.mode == "configure":
        from src.zone_config import ZoneConfigurator

        configurator = ZoneConfigurator(
            camera_index=args.camera,
            config_path=args.config,
        )
        zm = configurator.run()
        if zm.zones:
            if args.config:
                zm.save(args.config)
            else:
                zm.save()
            print(f"\nAuto-saved {len(zm.zones)} zone(s) on exit.")

    elif args.mode == "run":
        from src.run_mode import RunMode

        runner = RunMode(
            camera_index=args.camera,
            config_path=args.config,
            trims_path=args.trims,
            yolo_model=args.model,
            confidence=args.conf,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            collect_stats=args.stats,
            stats_csv=args.stats_csv,
        )
        runner.run()

    elif args.mode == "init-trims":
        from src.trim_config import create_sample_trims

        if args.trims:
            create_sample_trims(args.trims)
        else:
            create_sample_trims()

    elif args.mode == "simulate":
        from src.benchmark import Benchmark

        broker = args.mqtt_host or "localhost"
        bench = Benchmark(
            camera_index=0,
            num_frames=args.frames,
            broker_host=broker,
            broker_port=args.mqtt_port,
            config_path=args.config,
            yolo_model=args.model,
            confidence=args.conf,
            output_csv=args.output,
            use_synthetic=True,
        )
        bench.run()

    elif args.mode == "benchmark":
        from src.benchmark import Benchmark

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

    elif args.mode == "benchmark-server":
        from src.benchmark_server import CloudSimServer

        server = CloudSimServer(
            broker_host=args.mqtt_host,
            broker_port=args.mqtt_port,
            config_path=args.config,
            yolo_model=args.model,
            confidence=args.conf,
        )
        server.run()

    elif args.mode == "resource-monitor":
        from src.resource_monitor import main as rm_main
        # re-inject parsed args so resource_monitor doesn't re-parse
        sys.argv = [
            "resource_monitor",
            "--duration", str(args.duration),
            "--interval", str(args.interval),
            "--output", args.output,
        ]
        rm_main()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
