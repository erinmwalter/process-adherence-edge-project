import time

import cv2
import numpy as np

ALERT_DURATION = 2.0
FLASH_INTERVAL = 0.15


class AlertSystem:
    """Manages on-screen deviation alerts."""

    def __init__(self, duration: float = ALERT_DURATION) -> None:
        self.duration = duration
        self._alert_start: float = 0.0
        self._alert_message: str = ""
        self._active: bool = False

    def trigger(self, message: str) -> None:
        """Trigger a new alert with the given message."""
        self._alert_message = message
        self._alert_start = time.time()
        self._active = True

    @property
    def is_active(self) -> bool:
        if self._active and (time.time() - self._alert_start) > self.duration:
            self._active = False
        return self._active

    def draw(self, frame: np.ndarray) -> None:
        """Draw the alert overlay on the frame if active."""
        if not self.is_active:
            return

        h, w = frame.shape[:2]
        elapsed = time.time() - self._alert_start

        # flash on/off
        flash_on = int(elapsed / FLASH_INTERVAL) % 2 == 0

        if flash_on:
            # red border
            border = 12
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border)
            # red tinted overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # warning text — always visible during alert
        text_lines = ["!! DEVIATION !!", self._alert_message]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(text_lines):
            scale = 1.4 if i == 0 else 0.8
            thickness = 3 if i == 0 else 2
            text_size = cv2.getTextSize(line, font, scale, thickness)[0]
            tx = (w - text_size[0]) // 2
            ty = (h // 2) - 30 + i * 60

            # black outline for readability
            cv2.putText(frame, line, (tx, ty), font, scale, (0, 0, 0), thickness + 3)
            # white text
            cv2.putText(frame, line, (tx, ty), font, scale, (255, 255, 255), thickness)
