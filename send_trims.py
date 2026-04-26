"""
Send random trim commands to the edge via MQTT.
Usage: python3 send_trims.py
"""

import json
import time
import random
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT   = 1883
TOPIC  = "process_adherence/trim"
COUNT  = 30


def main():
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()
    time.sleep(0.3)

    for i in range(1, COUNT + 1):
        trim = random.choice(["A", "B", "C"])
        client.publish(TOPIC, json.dumps({"trim": trim}))
        print(f"  [{i:02d}] → trim={trim}")
        time.sleep(0.1)

    client.loop_stop()
    client.disconnect()
    print(f"\nDone — {COUNT} trim commands sent to {TOPIC}")


if __name__ == "__main__":
    main()
