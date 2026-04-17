"""
Database layer — stores cycle data in SQLite.

Tables:
  cycles: one row per completed cycle
  steps:  one row per pick step within a cycle (FK → cycles.id)
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "cycles.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cycles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trim_level      TEXT    NOT NULL,
            expected_seq    TEXT    NOT NULL,   -- JSON array
            observed_seq    TEXT    NOT NULL,   -- JSON array
            cycle_time_s    REAL    NOT NULL,
            completed       INTEGER NOT NULL,   -- 0/1
            error_count     INTEGER NOT NULL,
            received_at     TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS steps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id    INTEGER NOT NULL REFERENCES cycles(id),
            step_index  INTEGER NOT NULL,
            zone        TEXT    NOT NULL,
            expected    TEXT,
            correct     INTEGER NOT NULL,  -- 0/1
            timestamp   REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cycles_trim ON cycles(trim_level);
        CREATE INDEX IF NOT EXISTS idx_cycles_received ON cycles(received_at);
        CREATE INDEX IF NOT EXISTS idx_steps_cycle ON steps(cycle_id);
    """)
    conn.commit()
    conn.close()


def insert_cycle(cycle_data: dict, db_path: str = DB_PATH) -> int:
    """Insert a cycle + its steps. Returns the new cycle id."""
    conn = get_connection(db_path)
    cur = conn.execute(
        """INSERT INTO cycles (trim_level, expected_seq, observed_seq,
                               cycle_time_s, completed, error_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            cycle_data["trim_level"],
            json.dumps(cycle_data["expected_sequence"]),
            json.dumps(cycle_data["observed_sequence"]),
            cycle_data.get("cycle_time_s", 0),
            1 if cycle_data.get("completed") else 0,
            cycle_data.get("error_count", 0),
        ),
    )
    cycle_id = cur.lastrowid

    for i, step in enumerate(cycle_data.get("steps", [])):
        conn.execute(
            """INSERT INTO steps (cycle_id, step_index, zone, expected, correct, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                cycle_id,
                i,
                step["zone"],
                step.get("expected"),
                1 if step.get("correct") else 0,
                step.get("timestamp", 0),
            ),
        )

    conn.commit()
    conn.close()
    return cycle_id


# ── query helpers ────────────────────────────────────────────────

def get_all_cycles(limit: int = 200, db_path: str = DB_PATH) -> List[Dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM cycles ORDER BY received_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_cycle_with_steps(cycle_id: int, db_path: str = DB_PATH) -> Optional[Dict]:
    conn = get_connection(db_path)
    cycle = conn.execute("SELECT * FROM cycles WHERE id = ?", (cycle_id,)).fetchone()
    if cycle is None:
        conn.close()
        return None
    steps = conn.execute(
        "SELECT * FROM steps WHERE cycle_id = ? ORDER BY step_index", (cycle_id,)
    ).fetchall()
    conn.close()
    result = dict(cycle)
    result["steps"] = [dict(s) for s in steps]
    return result


def get_summary_stats(db_path: str = DB_PATH) -> Dict:
    """Aggregate stats for the dashboard."""
    conn = get_connection(db_path)

    total = conn.execute("SELECT COUNT(*) as n FROM cycles").fetchone()["n"]
    completed = conn.execute(
        "SELECT COUNT(*) as n FROM cycles WHERE completed = 1"
    ).fetchone()["n"]
    total_errors = conn.execute(
        "SELECT COALESCE(SUM(error_count), 0) as n FROM cycles"
    ).fetchone()["n"]
    avg_cycle = conn.execute(
        "SELECT COALESCE(AVG(cycle_time_s), 0) as v FROM cycles WHERE completed = 1"
    ).fetchone()["v"]

    # per-trim breakdown
    trim_rows = conn.execute("""
        SELECT trim_level,
               COUNT(*) as total,
               SUM(CASE WHEN error_count = 0 THEN 1 ELSE 0 END) as perfect,
               SUM(error_count) as errors,
               AVG(cycle_time_s) as avg_time
        FROM cycles
        GROUP BY trim_level
        ORDER BY trim_level
    """).fetchall()

    # recent error rate (last 50 cycles)
    recent = conn.execute("""
        SELECT COALESCE(AVG(CASE WHEN error_count > 0 THEN 1.0 ELSE 0.0 END), 0) as rate
        FROM (SELECT error_count FROM cycles ORDER BY received_at DESC LIMIT 50)
    """).fetchone()["rate"]

    conn.close()

    return {
        "total_cycles": total,
        "completed_cycles": completed,
        "total_errors": total_errors,
        "avg_cycle_time_s": round(avg_cycle, 2),
        "recent_error_rate": round(recent, 4),
        "by_trim": [dict(r) for r in trim_rows],
    }
