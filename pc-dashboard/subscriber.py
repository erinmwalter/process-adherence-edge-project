"""
MQTT subscriber — listens for cycle data from the edge device and stores it.

Subscribes to: process_adherence/cycle
On message: parse JSON → insert into SQLite via db.insert_cycle()
"""

import json
import paho.mqtt.client as mqtt

from db import insert_cycle


def create_subscriber(
    broker_host: str = "localhost",
    broker_port: int = 1883,
    topic: str = "process_adherence/cycle",
) -> mqtt.Client:
    client = mqtt.Client(protocol=mqtt.MQTTv311)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(topic)
            print(f"[MQTT] Connected & subscribed to '{topic}'")
        else:
            print(f"[MQTT] Connection failed, rc={rc}")

    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            cycle_id = insert_cycle(data)
            trim = data.get("trim_level", "?")
            errors = data.get("error_count", 0)
            ct = data.get("cycle_time_s", 0)
            print(f"[MQTT] Saved cycle #{cycle_id}: trim={trim}, "
                  f"errors={errors}, time={ct:.1f}s")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[MQTT] Bad message: {e}")
        except Exception as e:
            print(f"[MQTT] Error saving cycle: {e}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_host, broker_port, keepalive=60)
    return client
