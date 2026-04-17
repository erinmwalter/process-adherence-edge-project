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

    # ── init-trims sub-command ───────────────────────────────────
    trims_parser = subparsers.add_parser(
        "init-trims", help="Generate a sample trims.json config for testing."
    )
    trims_parser.add_argument(
        "--trims", type=str, default=None,
        help="Output path (default: config/trims.json)",
    )

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
        )
        runner.run()

    elif args.mode == "init-trims":
        from src.trim_config import create_sample_trims

        if args.trims:
            create_sample_trims(args.trims)
        else:
            create_sample_trims()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
