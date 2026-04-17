#!/usr/bin/env python3
"""
Line simulator — publishes random trim messages to MQTT at intervals,
simulating vehicles arriving on the manufacturing line.

Usage:
  python line_simulator.py [--host localhost] [--port 1883]
                           [--trims config/trims.json]
                           [--interval 15] [--count 0]

  --interval  seconds between vehicles (default: 15)
  --count     number of vehicles to simulate, 0 = infinite (default: 0)
"""

import argparse
import json
import random
import time

import paho.mqtt.client as mqtt


def load_trim_levels(path: str) -> list[str]:
    with open(path, "r") as fh:
        data = json.load(fh)
    return list(data.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Line Simulator — send trim messages")
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--trims", default="config/trims.json",
                        help="Path to trims.json to read available trim levels")
    parser.add_argument("--interval", type=float, default=15,
                        help="Seconds between vehicles (default: 15)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of vehicles to send, 0 = infinite (default: 0)")
    parser.add_argument("--topic", default="process_adherence/trim",
                        help="MQTT topic to publish to")
    args = parser.parse_args()

    # load available trims
    trim_levels = load_trim_levels(args.trims)
    if not trim_levels:
        print("ERROR: No trims found in config file.")
        return
    print(f"Loaded {len(trim_levels)} trim level(s): {', '.join(trim_levels)}")

    # connect to broker
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()
    print(f"Connected to MQTT broker at {args.host}:{args.port}")
    print(f"Publishing to topic: {args.topic}")
    print(f"Interval: {args.interval}s | Count: {'infinite' if args.count == 0 else args.count}")
    print("─" * 50)

    vehicle_num = 0
    try:
        while True:
            vehicle_num += 1
            trim = random.choice(trim_levels)
            payload = json.dumps({"trim": trim})
            client.publish(args.topic, payload)
            print(f"  Vehicle #{vehicle_num}: trim={trim}  [{time.strftime('%H:%M:%S')}]")

            if args.count > 0 and vehicle_num >= args.count:
                print(f"\nDone — sent {vehicle_num} vehicle(s).")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nStopped after {vehicle_num} vehicle(s).")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
