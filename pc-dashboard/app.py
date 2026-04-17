"""
Flask web app — serves the dashboard frontend and JSON API endpoints.

Routes:
  GET  /                  → dashboard HTML page
  GET  /api/summary       → aggregate stats
  GET  /api/cycles        → recent cycles list
  GET  /api/cycles/<id>   → single cycle with steps
"""

import json
from flask import Flask, jsonify, render_template, request

from db import get_all_cycles, get_cycle_with_steps, get_summary_stats

app = Flask(__name__, template_folder="templates", static_folder="static")


# ── HTML ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


# ── JSON API ─────────────────────────────────────────────────────

@app.route("/api/summary")
def api_summary():
    return jsonify(get_summary_stats())


@app.route("/api/cycles")
def api_cycles():
    limit = request.args.get("limit", 200, type=int)
    limit = min(max(limit, 1), 1000)
    cycles = get_all_cycles(limit=limit)
    # parse JSON strings back into lists for the API response
    for c in cycles:
        c["expected_seq"] = json.loads(c["expected_seq"])
        c["observed_seq"] = json.loads(c["observed_seq"])
    return jsonify(cycles)


@app.route("/api/cycles/<int:cycle_id>")
def api_cycle_detail(cycle_id: int):
    cycle = get_cycle_with_steps(cycle_id)
    if cycle is None:
        return jsonify({"error": "cycle not found"}), 404
    cycle["expected_seq"] = json.loads(cycle["expected_seq"])
    cycle["observed_seq"] = json.loads(cycle["observed_seq"])
    return jsonify(cycle)
