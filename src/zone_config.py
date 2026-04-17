"""
Zone configuration mode — interactive OpenCV UI for drawing zone polygons.

Controls:
  - Left-click   : add a vertex to the current polygon
  - Right-click   : undo last vertex
  - Enter / Return: finish current zone → prompts for name & type in terminal
  - 'c'           : clear all zones and start over
  - 's'           : save zones to disk
  - 'q' / Escape  : quit configuration mode
"""

import cv2
import numpy as np
from typing import List

from .zone_manager import ZoneManager, ZONE_TYPES, ZONE_COLORS

WINDOW_NAME = "Zone Configuration"


class ZoneConfigurator:
    def __init__(self, camera_index: int = 0, config_path: str | None = None) -> None:
        self.camera_index = camera_index
        self.zone_manager = ZoneManager()
        self.config_path = config_path
        self._current_polygon: List[List[int]] = []
        self._frame: np.ndarray | None = None

    # ── mouse callback ───────────────────────────────────────────

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_polygon.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and self._current_polygon:
            self._current_polygon.pop()

    # ── helpers ──────────────────────────────────────────────────

    def _draw_current_polygon(self, frame: np.ndarray) -> None:
        if not self._current_polygon:
            return
        pts = np.array(self._current_polygon, dtype=np.int32)
        for pt in pts:
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    @staticmethod
    def _prompt_zone_details() -> tuple[str, str]:
        print("\n── New zone ──")
        name = input("  Zone name (e.g. A, B, C): ").strip()
        if not name:
            name = f"Zone_{len(name)+1}"
        print(f"  Zone types: {', '.join(f'{i}: {t}' for i, t in enumerate(ZONE_TYPES))}")
        while True:
            choice = input("  Select zone type number: ").strip()
            if choice.isdigit() and 0 <= int(choice) < len(ZONE_TYPES):
                zone_type = ZONE_TYPES[int(choice)]
                break
            print("  Invalid choice, try again.")
        return name, zone_type

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw instruction overlay."""
        lines = [
            "LEFT CLICK: add point | RIGHT CLICK: undo point",
            "ENTER: finish zone | C: clear all | S: save | Q: quit",
            f"Zones defined: {len(self.zone_manager.zones)}  |  "
            f"Current polygon pts: {len(self._current_polygon)}",
        ]
        for i, line in enumerate(lines):
            y = 25 + i * 25
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
            )
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

    # ── main loop ────────────────────────────────────────────────

    def run(self) -> ZoneManager:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_cb)

        print("\n=== ZONE CONFIGURATION MODE ===")
        print("Click to draw polygon vertices on the camera feed.")
        print("Press ENTER when a polygon is done, then name it in the terminal.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed, retrying...")
                    continue

                # draw saved zones
                display = frame.copy()
                self.zone_manager.draw_zones(display)

                # draw in-progress polygon
                self._draw_current_polygon(display)

                # HUD
                self._draw_hud(display)

                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # q or Escape
                    break

                elif key == 13:  # Enter — finish current polygon
                    if len(self._current_polygon) < 3:
                        print("Need at least 3 points to define a zone.")
                        continue
                    name, zone_type = self._prompt_zone_details()
                    self.zone_manager.add_zone(name, zone_type, self._current_polygon)
                    print(f"  ✓ Zone '{name}' ({zone_type}) added with "
                          f"{len(self._current_polygon)} vertices.")
                    self._current_polygon = []

                elif key == ord("c"):
                    self.zone_manager.clear()
                    self._current_polygon = []
                    print("All zones cleared.")

                elif key == ord("s"):
                    save_path = self.config_path or None
                    if save_path:
                        self.zone_manager.save(save_path)
                    else:
                        self.zone_manager.save()
                    print(f"Zones saved ({len(self.zone_manager.zones)} zones).")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return self.zone_manager
