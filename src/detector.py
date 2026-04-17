import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ultralytics import YOLO

# COCO pose keypoint indices for wrists
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10


# ── data classes ─────────────────────────────────────────────────

@dataclass
class WristDetection:
    """A single detected wrist position."""
    x: int
    y: int
    confidence: float
    side: str  # "left" | "right"
    person_id: int  # which person this wrist belongs to


@dataclass
class PoseDetection:
    """A detected person with their wrist positions."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    left_wrist: Optional[WristDetection] = None
    right_wrist: Optional[WristDetection] = None

    @property
    def wrists(self) -> List[WristDetection]:
        return [w for w in (self.left_wrist, self.right_wrist) if w is not None]


@dataclass
class PartDetection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    color_name: str                  # "red" | "blue" | "green"
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


# ── HSV ranges for colour detection ─────────────────────────────
# Tuned for bright balls/objects under typical indoor lighting.
# Red wraps around the hue circle, so we use two ranges.

COLOR_RANGES = {
    "red": [
        {"lower": np.array([0, 100, 100]),   "upper": np.array([10, 255, 255])},
        {"lower": np.array([160, 100, 100]), "upper": np.array([180, 255, 255])},
    ],
    "green": [
        {"lower": np.array([35, 80, 80]),  "upper": np.array([85, 255, 255])},
    ],
    "blue": [
        {"lower": np.array([90, 80, 80]),  "upper": np.array([130, 255, 255])},
    ],
}

MIN_PART_AREA = 300  # minimum contour area (px²) to count as a valid part

PART_DRAW_COLORS = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
    "blue":  (255, 0, 0),
}

# Minimum keypoint confidence to consider a wrist "visible"
MIN_WRIST_CONF = 0.3


class Detector:
    """Wraps YOLOv8n-pose wrist extraction and HSV colour part detection."""

    def __init__(
        self,
        yolo_model: str = "yolov8n-pose.pt",
        confidence: float = 0.45,
    ):
        self.model = YOLO(yolo_model)
        self.confidence = confidence

    # ── pose / wrist detection ───────────────────────────────────

    def detect_poses(self, frame: np.ndarray) -> List[PoseDetection]:
        """Run pose estimation and extract wrist keypoints."""
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections: List[PoseDetection] = []

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            keypoints = r.keypoints.data  # (N, 17, 3) — x, y, conf
            for person_idx, (box, kps) in enumerate(zip(r.boxes, keypoints)):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                pose = PoseDetection(bbox=(x1, y1, x2, y2), confidence=conf)

                # left wrist
                lw = kps[LEFT_WRIST_IDX]
                if float(lw[2]) >= MIN_WRIST_CONF:
                    pose.left_wrist = WristDetection(
                        x=int(lw[0]), y=int(lw[1]),
                        confidence=float(lw[2]),
                        side="left", person_id=person_idx,
                    )

                # right wrist
                rw = kps[RIGHT_WRIST_IDX]
                if float(rw[2]) >= MIN_WRIST_CONF:
                    pose.right_wrist = WristDetection(
                        x=int(rw[0]), y=int(rw[1]),
                        confidence=float(rw[2]),
                        side="right", person_id=person_idx,
                    )

                detections.append(pose)

        return detections

    # ── colour-based part detection ──────────────────────────────

    @staticmethod
    def detect_colored_parts(
        frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> List[PartDetection]:
        """Detect red, blue, green objects. Optionally restrict search to a binary mask."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        detections: List[PartDetection] = []

        for color_name, ranges in COLOR_RANGES.items():
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in ranges:
                combined_mask |= cv2.inRange(hsv, r["lower"], r["upper"])

            if mask is not None:
                combined_mask = cv2.bitwise_and(combined_mask, mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_PART_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    PartDetection(bbox=(x, y, x + w, y + h), color_name=color_name)
                )

        return detections

    # ── drawing helpers ──────────────────────────────────────────

    @staticmethod
    def draw_poses(frame: np.ndarray, poses: List[PoseDetection]) -> None:
        for pose in poses:
            x1, y1, x2, y2 = pose.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {pose.confidence:.0%}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
            # draw wrist markers
            for w in pose.wrists:
                color = (0, 255, 255) if w.side == "left" else (255, 0, 255)
                cv2.circle(frame, (w.x, w.y), 8, color, -1)
                cv2.putText(
                    frame, f"{w.side[0].upper()}W",
                    (w.x + 10, w.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

    @staticmethod
    def draw_parts(frame: np.ndarray, parts: List[PartDetection]) -> None:
        for p in parts:
            x1, y1, x2, y2 = p.bbox
            color = PART_DRAW_COLORS.get(p.color_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, p.color_name, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
