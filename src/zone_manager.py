"""
Zone manager — handles zone definitions, persistence, and point-in-polygon checks.

Each zone is a dict:
{
    "name": "A",
    "zone_type": "pick_1",          # pick_1 | pick_2 | walking | working
    "polygon": [[x1,y1], [x2,y2], ...],
    "color": [B, G, R]              # display colour (BGR for OpenCV)
}
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

# Predefined palette so zones are visually distinct (BGR)
ZONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pick_1":  (0, 255, 255),   # yellow
    "pick_2":  (255, 0, 255),   # magenta
    "pick_3":  (255, 165, 0),   # blue-ish orange
    "pick_4":  (0, 255, 128),   # spring green
    "walking": (255, 255, 0),   # cyan
    "working": (0, 165, 255),   # orange
}

ZONE_TYPES = list(ZONE_COLORS.keys())

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "zones.json"
)


class ZoneManager:
    """Stores zone polygons and checks containment."""

    def __init__(self) -> None:
        self.zones: List[Dict] = []

    # ── persistence ──────────────────────────────────────────────

    def save(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.zones, fh, indent=2)

    def load(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        with open(path, "r") as fh:
            self.zones = json.load(fh)

    # ── zone CRUD ────────────────────────────────────────────────

    def add_zone(self, name: str, zone_type: str, polygon: List[List[int]]) -> None:
        if zone_type not in ZONE_TYPES:
            raise ValueError(f"zone_type must be one of {ZONE_TYPES}")
        if len(polygon) < 3:
            raise ValueError("A zone polygon needs at least 3 points")
        color = list(ZONE_COLORS[zone_type])
        self.zones.append(
            {"name": name, "zone_type": zone_type, "polygon": polygon, "color": color}
        )

    def clear(self) -> None:
        self.zones.clear()

    # ── spatial queries ──────────────────────────────────────────

    def point_in_zone(self, x: int, y: int, zone: Dict) -> bool:
        poly = np.array(zone["polygon"], dtype=np.int32)
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def get_zones_for_point(self, x: int, y: int) -> List[Dict]:
        return [z for z in self.zones if self.point_in_zone(x, y, z)]

    def get_zone_mask(self, zone: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Return a binary mask for the zone polygon over the given (H, W)."""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        poly = np.array(zone["polygon"], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        return mask

    # ── drawing helpers ──────────────────────────────────────────

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        overlay = frame.copy()
        for zone in self.zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            color = tuple(zone["color"])
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            # label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            label = f'{zone["name"]} ({zone["zone_type"]})'
            cv2.putText(
                frame, label, (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
