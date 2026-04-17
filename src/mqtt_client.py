"""
MQTT client — handles communication between edge device and PC.

Subscribes to:
  - {topic_prefix}/trim   → receives trim level for the next cycle
                             payload: {"trim": "A"}

Publishes to:
  - {topic_prefix}/cycle  → sends cycle result data after each cycle
                             payload: CycleResult.to_dict()

Uses paho-mqtt. Broker address and topics are configurable.
"""

import json
import threading
from typing import Callable, Optional

import paho.mqtt.client as mqtt


class MQTTClient:
    """Thin wrapper around paho-mqtt for the process adherence system."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "process_adherence",
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix

        self._trim_topic = f"{topic_prefix}/trim"
        self._cycle_topic = f"{topic_prefix}/cycle"

        self._client = mqtt.Client(protocol=mqtt.MQTTv311)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._trim_callback: Optional[Callable[[str], None]] = None
        self._connected = False

    # ── connection ───────────────────────────────────────────────

    def connect(self) -> None:
        self._client.connect(self.broker_host, self.broker_port, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            client.subscribe(self._trim_topic)
            print(f"MQTT connected to {self.broker_host}:{self.broker_port}")
            print(f"  Subscribed to: {self._trim_topic}")
        else:
            print(f"MQTT connection failed, rc={rc}")

    # ── receiving trim messages ──────────────────────────────────

    def set_trim_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback that receives the trim level string."""
        self._trim_callback = callback

    def _on_message(self, client, userdata, msg) -> None:
        if msg.topic == self._trim_topic:
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                trim_level = payload.get("trim", "")
                if trim_level and self._trim_callback:
                    self._trim_callback(trim_level)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"MQTT: invalid trim message: {e}")

    # ── publishing cycle data ────────────────────────────────────

    def publish_cycle(self, cycle_data: dict) -> None:
        """Publish cycle result to the cycle topic."""
        payload = json.dumps(cycle_data)
        self._client.publish(self._cycle_topic, payload)

    @property
    def is_connected(self) -> bool:
        return self._connected


class DummyMQTTClient:
    """
    Stand-in when no MQTT broker is available.
    Allows the system to run in offline / demo mode.
    """

    def __init__(self) -> None:
        self._trim_callback: Optional[Callable[[str], None]] = None

    def connect(self) -> None:
        print("MQTT disabled — running in offline mode.")

    def disconnect(self) -> None:
        pass

    def set_trim_callback(self, callback: Callable[[str], None]) -> None:
        self._trim_callback = callback

    def publish_cycle(self, cycle_data: dict) -> None:
        print(f"[OFFLINE] Cycle data: {json.dumps(cycle_data, indent=2)}")

    def inject_trim(self, trim_level: str) -> None:
        """Manually inject a trim level (for testing without a broker)."""
        if self._trim_callback:
            self._trim_callback(trim_level)

    @property
    def is_connected(self) -> bool:
        return False
