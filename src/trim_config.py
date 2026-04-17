"""
Trim configuration — maps trim levels to expected pick sequences.

Each trim config defines an ordered list of zone names the operator must
pick from.  For example, trim "A" might require: pick from zone "BIN_1",
then "BIN_3", then "BIN_2".

Config is stored in a JSON file (config/trims.json) like:
{
  "A": ["BIN_1", "BIN_3", "BIN_2"],
  "B": ["BIN_2", "BIN_1"],
  "C": ["BIN_1", "BIN_2", "BIN_3"]
}
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_TRIMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "trims.json"
)


class TrimConfig:
    """Loads and queries trim-level → expected zone-pick sequences."""

    def __init__(self) -> None:
        self.trims: Dict[str, List[str]] = {}

    def save(self, path: str = DEFAULT_TRIMS_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.trims, fh, indent=2)

    def load(self, path: str = DEFAULT_TRIMS_PATH) -> None:
        with open(path, "r") as fh:
            self.trims = json.load(fh)

    def get_sequence(self, trim_level: str) -> Optional[List[str]]:
        """Return the expected pick-zone sequence for a trim level, or None."""
        return self.trims.get(trim_level)

    def list_trims(self) -> List[str]:
        return list(self.trims.keys())

    def set_sequence(self, trim_level: str, sequence: List[str]) -> None:
        self.trims[trim_level] = sequence


def create_sample_trims(path: str = DEFAULT_TRIMS_PATH) -> None:
    """Write a sample trims.json for testing with three configs."""
    cfg = TrimConfig()
    cfg.set_sequence("A", ["BIN_1", "BIN_3", "BIN_2"])
    cfg.set_sequence("B", ["BIN_2", "BIN_1"])
    cfg.set_sequence("C", ["BIN_1", "BIN_2", "BIN_3"])
    cfg.save(path)
    print(f"Sample trims config written to {path}")
