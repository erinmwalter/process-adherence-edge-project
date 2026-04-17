/* ── Dashboard JS — fetches data and renders tables ── */

const REFRESH_MS = 5000;

// ── Fetch helpers ───────────────────────────────────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Render summary cards ────────────────────────────────────────

async function loadSummary() {
  try {
    const s = await fetchJSON("/api/summary");
    document.getElementById("total-cycles").textContent = s.total_cycles;
    document.getElementById("completed-cycles").textContent = s.completed_cycles;
    document.getElementById("total-errors").textContent = s.total_errors;
    document.getElementById("avg-cycle-time").textContent = s.avg_cycle_time_s + "s";
    document.getElementById("recent-error-rate").textContent =
      (s.recent_error_rate * 100).toFixed(1) + "%";

    // per-trim table
    const tbody = document.getElementById("trim-tbody");
    tbody.innerHTML = "";
    for (const t of s.by_trim) {
      const rate = t.total > 0 ? ((t.perfect / t.total) * 100).toFixed(1) : "—";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${esc(t.trim_level)}</strong></td>
        <td>${t.total}</td>
        <td>${t.perfect}</td>
        <td>${t.errors}</td>
        <td>${t.avg_time !== null ? t.avg_time.toFixed(1) : "—"}</td>
        <td>
          <div class="bar-bg">
            <div class="bar-fill ${Number(rate) >= 80 ? 'bar-green' : 'bar-red'}"
                 style="width:${rate === "—" ? 0 : rate}%"></div>
          </div>
          <span style="font-size:0.75rem;color:#888">${rate}%</span>
        </td>`;
      tbody.appendChild(tr);
    }
  } catch (e) {
    console.error("summary fetch failed:", e);
  }
}

// ── Render recent cycles table ──────────────────────────────────

async function loadCycles() {
  try {
    const cycles = await fetchJSON("/api/cycles?limit=100");
    const tbody = document.getElementById("cycles-tbody");
    tbody.innerHTML = "";
    for (const c of cycles) {
      const tr = document.createElement("tr");
      tr.onclick = () => openCycleDetail(c.id);

      let badge;
      if (c.error_count === 0 && c.completed) {
        badge = '<span class="badge badge-ok">OK</span>';
      } else if (c.error_count > 0) {
        badge = '<span class="badge badge-err">ERROR</span>';
      } else {
        badge = '<span class="badge badge-partial">INCOMPLETE</span>';
      }

      tr.innerHTML = `
        <td>${c.id}</td>
        <td style="font-size:0.78rem;color:#888">${esc(c.received_at)}</td>
        <td><strong>${esc(c.trim_level)}</strong></td>
        <td class="seq">${esc(c.expected_seq.join(" → "))}</td>
        <td class="seq">${esc(c.observed_seq.join(" → "))}</td>
        <td>${c.cycle_time_s.toFixed(1)}s</td>
        <td style="color:${c.error_count ? '#ef5350' : '#4caf50'}">${c.error_count}</td>
        <td>${badge}</td>`;
      tbody.appendChild(tr);
    }
  } catch (e) {
    console.error("cycles fetch failed:", e);
  }
}

// ── Cycle detail modal ──────────────────────────────────────────

async function openCycleDetail(id) {
  try {
    const c = await fetchJSON(`/api/cycles/${id}`);
    const body = document.getElementById("modal-body");

    let html = `
      <p><strong>Cycle #${c.id}</strong> &middot; Trim <strong>${esc(c.trim_level)}</strong>
         &middot; ${c.cycle_time_s.toFixed(1)}s
         &middot; ${c.error_count} error(s)</p>
      <p style="margin-top:0.5rem;font-size:0.82rem;color:#888">
        Expected: <span class="seq">${esc(c.expected_seq.join(" → "))}</span><br>
        Observed: <span class="seq">${esc(c.observed_seq.join(" → "))}</span>
      </p>
      <h3 style="margin-top:1rem;font-size:0.9rem;color:#aaa">Steps</h3>`;

    for (const s of c.steps) {
      const cls = s.correct ? "step-ok" : "step-bad";
      const icon = s.correct ? "✓" : "✗";
      const exp = s.expected ? ` (expected ${esc(s.expected)})` : "";
      html += `
        <div class="step-row">
          <span class="step-idx">${s.step_index + 1}.</span>
          <span class="step-zone ${cls}">${icon} ${esc(s.zone)}${s.correct ? "" : exp}</span>
        </div>`;
    }

    body.innerHTML = html;
    document.getElementById("modal-overlay").classList.remove("hidden");
  } catch (e) {
    console.error("detail fetch failed:", e);
  }
}

function closeModal() {
  document.getElementById("modal-overlay").classList.add("hidden");
}

// ── Utility ─────────────────────────────────────────────────────

function esc(str) {
  const d = document.createElement("div");
  d.textContent = String(str);
  return d.innerHTML;
}

// ── Init & auto-refresh ─────────────────────────────────────────

async function refresh() {
  await Promise.all([loadSummary(), loadCycles()]);
}

refresh();
setInterval(refresh, REFRESH_MS);
