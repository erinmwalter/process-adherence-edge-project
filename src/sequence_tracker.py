"""
Sequence tracker — tracks wrist zone entries and compares against expected order.

Core logic:
  1. Each frame, the wrist (x, y) is checked against all defined zones.
  2. When a wrist enters a new zone (that it wasn't in the previous frame),
     a "zone entry" event is recorded.
  3. The sequence of entered zones is compared step-by-step against the
     expected sequence for the current trim level.
  4. If a deviation is found, a deviation event is raised.

Debounce: a zone entry only counts if the wrist stays in the zone for
at least DEBOUNCE_FRAMES consecutive frames (avoids flickering at edges).
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict

from .zone_manager import ZoneManager


DEBOUNCE_FRAMES = 5  # wrist must be in zone for N frames to count as entry


@dataclass
class StepEvent:
    """A single observed pick step."""
    zone_name: str
    timestamp: float
    expected_zone: Optional[str] = None
    is_correct: bool = True


@dataclass
class CycleResult:
    """Summary of a completed (or aborted) cycle."""
    trim_level: str
    expected_sequence: List[str]
    observed_sequence: List[str]
    steps: List[StepEvent]
    start_time: float
    end_time: float = 0.0
    completed: bool = False
    error_count: int = 0

    @property
    def cycle_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    def to_dict(self) -> Dict:
        return {
            "trim_level": self.trim_level,
            "expected_sequence": self.expected_sequence,
            "observed_sequence": self.observed_sequence,
            "steps": [
                {
                    "zone": s.zone_name,
                    "expected": s.expected_zone,
                    "correct": s.is_correct,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "cycle_time_s": round(self.cycle_time, 3),
            "completed": self.completed,
            "error_count": self.error_count,
        }


class SequenceTracker:
    """Tracks wrist zone entries and validates against an expected sequence."""

    def __init__(self, zone_manager: ZoneManager) -> None:
        self.zone_manager = zone_manager

        # current cycle state
        self._expected_sequence: List[str] = []
        self._trim_level: str = ""
        self._observed: List[str] = []
        self._steps: List[StepEvent] = []
        self._cycle_start: float = 0.0
        self._error_count: int = 0
        self._active: bool = False

        # debounce state: for each wrist, track consecutive frames in a zone
        self._prev_zone: Optional[str] = None
        self._zone_frame_count: int = 0
        self._pending_zone: Optional[str] = None

        # latest deviation info (consumed by alert system)
        self.last_deviation: Optional[StepEvent] = None

    # ── cycle management ─────────────────────────────────────────

    def start_cycle(self, trim_level: str, expected_sequence: List[str]) -> None:
        self._expected_sequence = list(expected_sequence)
        self._trim_level = trim_level
        self._observed = []
        self._steps = []
        self._cycle_start = time.time()
        self._error_count = 0
        self._active = True
        self._prev_zone = None
        self._zone_frame_count = 0
        self._pending_zone = None
        self.last_deviation = None

    def end_cycle(self) -> Optional[CycleResult]:
        if not self._active:
            return None
        self._active = False
        result = CycleResult(
            trim_level=self._trim_level,
            expected_sequence=self._expected_sequence,
            observed_sequence=self._observed,
            steps=self._steps,
            start_time=self._cycle_start,
            end_time=time.time(),
            completed=(self._observed == self._expected_sequence),
            error_count=self._error_count,
        )
        return result

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def current_step_index(self) -> int:
        return len(self._observed)

    @property
    def expected_next_zone(self) -> Optional[str]:
        idx = self.current_step_index
        if idx < len(self._expected_sequence):
            return self._expected_sequence[idx]
        return None

    @property
    def is_sequence_complete(self) -> bool:
        return len(self._observed) >= len(self._expected_sequence)

    @property
    def observed_sequence(self) -> List[str]:
        return list(self._observed)

    @property
    def expected_sequence(self) -> List[str]:
        return list(self._expected_sequence)

    @property
    def trim_level(self) -> str:
        return self._trim_level

    @property
    def error_count(self) -> int:
        return self._error_count

    # ── per-frame update ─────────────────────────────────────────

    def update(self, wrist_x: int, wrist_y: int) -> Optional[StepEvent]:
        """
        Called each frame with the dominant wrist position.
        Returns a StepEvent if a new zone entry was recorded, else None.
        """
        if not self._active:
            return None

        # find which zone (if any) the wrist is in (only pick-type zones)
        zones_hit = self.zone_manager.get_zones_for_point(wrist_x, wrist_y)
        # filter to only pick zones for sequence tracking
        tracked_zones = [z for z in zones_hit if z["zone_type"].startswith("pick")
                         or z["zone_type"] in ("walking", "working")]
        current_zone = tracked_zones[0]["name"] if tracked_zones else None

        # debounce logic
        if current_zone == self._pending_zone:
            self._zone_frame_count += 1
        else:
            self._pending_zone = current_zone
            self._zone_frame_count = 1

        # only register entry after debounce threshold and if it's a new zone
        if (
            current_zone is not None
            and current_zone != self._prev_zone
            and self._zone_frame_count >= DEBOUNCE_FRAMES
        ):
            self._prev_zone = current_zone
            return self._record_zone_entry(current_zone)

        # if wrist left all zones, reset prev_zone so re-entry counts
        if current_zone is None and self._zone_frame_count >= DEBOUNCE_FRAMES:
            self._prev_zone = None

        return None

    def _record_zone_entry(self, zone_name: str) -> StepEvent:
        """Record a zone entry and check against expected sequence."""
        expected = self.expected_next_zone
        is_correct = (zone_name == expected)

        step = StepEvent(
            zone_name=zone_name,
            timestamp=time.time(),
            expected_zone=expected,
            is_correct=is_correct,
        )

        self._observed.append(zone_name)
        self._steps.append(step)

        if not is_correct:
            self._error_count += 1
            self.last_deviation = step

        return step
