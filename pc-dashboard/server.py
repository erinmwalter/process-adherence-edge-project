#!/usr/bin/env python3
"""
PC Dashboard — MQTT subscriber + Flask web dashboard.

Runs two things in one process:
  1. MQTT subscriber thread — listens to process_adherence/cycle and writes to SQLite
  2. Flask web server — serves the dashboard at http://localhost:5000

Usage:
  python server.py [--mqtt-host localhost] [--mqtt-port 1883] [--port 5000]
"""

import argparse
import threading

from db import init_db
from subscriber import create_subscriber
from app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="PC Dashboard — MQTT + Web")
    parser.add_argument(
        "--mqtt-host", type=str, default="localhost",
        help="MQTT broker hostname (default: localhost)",
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883,
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Web dashboard port (default: 5000)",
    )
    parser.add_argument(
        "--no-mqtt", action="store_true",
        help="Disable MQTT (web-only mode, useful if broker is not running)",
    )
    args = parser.parse_args()

    # init database
    init_db()
    print("[DB] SQLite database initialized.")

    # start MQTT subscriber in background thread
    if not args.no_mqtt:
        try:
            client = create_subscriber(
                broker_host=args.mqtt_host,
                broker_port=args.mqtt_port,
            )
            client.loop_start()
            print(f"[MQTT] Subscriber running (broker: {args.mqtt_host}:{args.mqtt_port})")
        except Exception as e:
            print(f"[MQTT] Could not connect to broker: {e}")
            print("[MQTT] Running in web-only mode.")
    else:
        print("[MQTT] Disabled via --no-mqtt flag.")

    # start Flask
    print(f"[WEB] Dashboard at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
