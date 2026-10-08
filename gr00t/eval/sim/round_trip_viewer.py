# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build self-contained HTML viewers for round-trip-consistency (C_i) rollouts.

VideoRecordingWrapper writes a ``<video stem>.json`` sidecar next to every kept
rollout video when round-trip checking is on (see its _save_round_trip_data):
the episode's action_c_i / latent_c_i per depth, one entry per recorded frame,
plus the env step each frame shows. This module turns a folder of those into:

* ``<video dir>/index.html`` - pick an episode, play its video next to the
  C_i-vs-step plots; a cursor follows playback, clicking a plot seeks the
  video, hovering reads out every visible series, and depths / metrics can be
  toggled on and off.
* ``<root>/index.html`` (with ``--root``) - one row per task folder, linking
  to each task's viewer, with success counts and mean C_i on failed vs.
  succeeded episodes.

The pages are static and offline: all C_i data is embedded in the HTML and
the videos are referenced by relative path, so the folder can be opened
directly (``file://``) or copied anywhere as a whole. Stdlib only, so it runs
in any venv:

    python gr00t/eval/sim/round_trip_viewer.py <video dir>        # one task
    python gr00t/eval/sim/round_trip_viewer.py <root> --root      # every task + index
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from statistics import mean


SIDECAR_FORMAT = "gr00t_round_trip_ci/v1"
VIEWER_NAME = "index.html"


def load_episodes(video_dir: Path) -> list[dict]:
    """All C_i sidecars in ``video_dir`` whose video still exists, sorted by
    video filename. Non-sidecar JSON files are ignored."""
    episodes = []
    for path in sorted(Path(video_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("format") != SIDECAR_FORMAT:
            continue
        if not (path.parent / data["video"]).exists():
            continue
        episodes.append(data)
    episodes.sort(key=lambda d: d["video"])
    return episodes


def episode_mean_action_ci(episode: dict) -> float | None:
    """Mean action C_i over every recorded frame and depth that has a value -
    the single number shown per episode in the list and the root index."""
    values = [v for column in episode["action_c_i"].values() for v in column if v is not None]
    return mean(values) if values else None


def _embed_json(payload) -> str:
    # Safe inside <script type="application/json">: a literal "</" could end
    # the script element early.
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def build_task_viewer(
    video_dir: Path, title: str | None = None, link_to_root: bool = False
) -> Path | None:
    """Write ``video_dir/index.html`` for the episodes in ``video_dir``.
    ``link_to_root`` adds an "All tasks" link to ``../index.html`` (set by
    build_root_index, which writes that page). Returns the viewer's path, or
    None if the folder has no C_i sidecars."""
    video_dir = Path(video_dir)
    episodes = load_episodes(video_dir)
    if not episodes:
        return None
    title = title or video_dir.name
    page = (
        _TASK_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__STYLE__", _STYLE)
        .replace(
            "__UPLINK__",
            f'<a href="../{VIEWER_NAME}">All tasks</a>' if link_to_root else "",
        )
        .replace("__DATA__", _embed_json({"title": title, "episodes": episodes}))
        .replace("__SCRIPT__", _TASK_SCRIPT)
    )
    out = video_dir / VIEWER_NAME
    out.write_text(page)
    return out


def build_root_index(root: Path, title: str | None = None) -> Path | None:
    """Build a viewer for every immediate subfolder of ``root`` that has C_i
    sidecars, then write ``root/index.html`` linking to them. Returns its
    path, or None if no subfolder had any."""
    root = Path(root)
    rows = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        episodes = load_episodes(task_dir)
        if not episodes:
            continue
        build_task_viewer(task_dir, link_to_root=True)
        fail = [episode_mean_action_ci(e) for e in episodes if not e["success"]]
        succ = [episode_mean_action_ci(e) for e in episodes if e["success"]]
        fail = [v for v in fail if v is not None]
        succ = [v for v in succ if v is not None]
        rows.append(
            {
                "task": task_dir.name,
                "href": f"{task_dir.name}/{VIEWER_NAME}",
                "n": len(episodes),
                "n_success": sum(1 for e in episodes if e["success"]),
                "mean_ci_fail": mean(fail) if fail else None,
                "mean_ci_succ": mean(succ) if succ else None,
            }
        )
    if not rows:
        return None
    title = title or root.name
    page = (
        _ROOT_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__STYLE__", _STYLE)
        .replace("__DATA__", _embed_json({"title": title, "tasks": rows}))
        .replace("__SCRIPT__", _ROOT_SCRIPT)
    )
    out = root / VIEWER_NAME
    out.write_text(page)
    return out


# ---------------------------------------------------------------------------
# Page assets. Colors are role tokens (light by default, dark via the OS
# setting or the theme toggle); one fixed hue per depth, assigned in order.
# ---------------------------------------------------------------------------

_STYLE = r"""
.viz-root {
  color-scheme: light;
  --surface-0: #f3f2ee; --surface-1: #fcfcfb; --border: #dddcd5;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #77766f;
  --grid: #e9e8e3; --axis: #b9b8b0; --cursor: #0b0b0b; --hover: #77766f;
  --good: #0ca30c; --critical: #d03b3b; --accent-bg: #e8f0fb;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --series-4: #eda100; --series-5: #e87ba4; --series-6: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
    --grid: #2b2b28; --axis: #55554f; --cursor: #ffffff; --hover: #94938a;
    --accent-bg: #1f2c3d;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --series-4: #c98500; --series-5: #d55181; --series-6: #008300;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
  --grid: #2b2b28; --axis: #55554f; --cursor: #ffffff; --hover: #94938a;
  --accent-bg: #1f2c3d;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --series-4: #c98500; --series-5: #d55181; --series-6: #008300;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body.viz-root {
  background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
header {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 16px;
  padding: 14px 16px; border-bottom: 1px solid var(--border); background: var(--surface-1);
}
header h1 { font-size: 16px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }
header .meta { color: var(--text-secondary); }
header .spacer { flex: 1; }
button, select {
  font: inherit; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; cursor: pointer;
}
button[aria-pressed="true"] { background: var(--accent-bg); border-color: var(--series-1); }
a { color: var(--series-1); }
.layout { display: grid; grid-template-columns: 260px minmax(0, 1fr); min-height: calc(100vh - 56px); }
@media (max-width: 860px) { .layout { grid-template-columns: minmax(0, 1fr); } }
aside { border-right: 1px solid var(--border); background: var(--surface-1); min-width: 0; }
@media (max-width: 860px) { aside { border-right: 0; border-bottom: 1px solid var(--border); } }
.filters { display: flex; gap: 6px; padding: 10px 12px; flex-wrap: wrap; }
.episodes { list-style: none; margin: 0; padding: 0 6px 12px; max-height: calc(100vh - 120px); overflow: auto; }
@media (max-width: 860px) { .episodes { max-height: 220px; } }
.episodes li button {
  width: 100%; display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center;
  text-align: left; border: 1px solid transparent; background: transparent; margin: 1px 0;
}
.episodes li button:hover { border-color: var(--border); }
.episodes li button[aria-current="true"] { background: var(--accent-bg); border-color: var(--series-1); }
.ep-index { color: var(--text-muted); font-variant-numeric: tabular-nums; }
.ep-ci { color: var(--text-secondary); font-variant-numeric: tabular-nums; font-size: 12px; }
.status { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: var(--text-secondary); }
.status svg { flex: none; }
main { padding: 14px 16px; min-width: 0; }
.stage { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
@media (max-width: 1200px) { .stage { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-width: 0; }
.panel h2 { font-size: 13px; font-weight: 600; margin: 0 0 8px; color: var(--text-secondary); }
video { width: 100%; display: block; border-radius: 4px; background: #000; }
.video-meta { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
.controls { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; margin-bottom: 10px; }
.controls .group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.controls .label { color: var(--text-muted); font-size: 12px; margin-right: 2px; }
.toggle { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
.chart { position: relative; margin-bottom: 6px; }
.chart svg { width: 100%; height: auto; display: block; touch-action: none; cursor: crosshair; }
.chart-title { font-size: 12px; color: var(--text-secondary); margin: 2px 0 0 4px; }
.tooltip {
  position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.15);
  white-space: nowrap; display: none; z-index: 2; font-variant-numeric: tabular-nums;
}
.tooltip .t-head { color: var(--text-muted); margin-bottom: 2px; }
.tooltip .t-row { display: flex; align-items: center; gap: 6px; }
.tooltip strong { font-weight: 600; }
table.readout { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin-top: 8px; }
table.readout th, table.readout td { padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: right; }
table.readout th:first-child, table.readout td:first-child { text-align: left; }
table.readout th { color: var(--text-muted); font-weight: 500; font-size: 12px; }
.hint { color: var(--text-muted); font-size: 12px; margin-top: 6px; }
table.tasks { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
table.tasks th, table.tasks td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right; }
table.tasks th:first-child, table.tasks td:first-child { text-align: left; overflow-wrap: anywhere; }
table.tasks th { color: var(--text-muted); font-weight: 500; font-size: 12px; }
.table-wrap { overflow-x: auto; }
.root-main { padding: 16px; max-width: 1100px; }
"""

_THEME_SCRIPT = r"""
(function () {
  const KEY = "rtc-viewer-theme";
  const root = document.documentElement;
  let stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) {}
  if (stored === "light" || stored === "dark") root.setAttribute("data-theme", stored);
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const label = () => {
    const t = root.getAttribute("data-theme");
    btn.textContent = t === "dark" ? "Theme: dark" : t === "light" ? "Theme: light" : "Theme: auto";
  };
  label();
  btn.addEventListener("click", () => {
    const t = root.getAttribute("data-theme");
    const next = t === null ? "light" : t === "light" ? "dark" : null;
    if (next) root.setAttribute("data-theme", next); else root.removeAttribute("data-theme");
    try { if (next) localStorage.setItem(KEY, next); else localStorage.removeItem(KEY); } catch (e) {}
    label();
    document.dispatchEvent(new Event("themechange"));
  });
})();
"""

_TASK_TEMPLATE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C_i viewer: __TITLE__</title>
<style>__STYLE__</style>
</head>
<body class="viz-root">
<header>
  <h1>__TITLE__</h1>
  <span class="meta" id="task-meta"></span>
  <span class="spacer"></span>
  __UPLINK__
  <button type="button" id="theme-toggle">Theme: auto</button>
</header>
<div class="layout">
  <aside>
    <div class="filters" role="group" aria-label="Filter episodes">
      <button type="button" data-filter="all" aria-pressed="true">All</button>
      <button type="button" data-filter="success" aria-pressed="false">Success</button>
      <button type="button" data-filter="failure" aria-pressed="false">Failure</button>
    </div>
    <ul class="episodes" id="episode-list" aria-label="Episodes"></ul>
  </aside>
  <main>
    <div class="stage">
      <section class="panel" aria-label="Rollout video">
        <h2 id="video-title">Video</h2>
        <video id="video" controls muted playsinline preload="auto"></video>
        <div class="video-meta">
          <span id="cur-step">step -</span>
          <span id="cur-frame">frame -</span>
          <span id="ep-status"></span>
        </div>
        <div class="hint">Click or drag on a plot to seek. Left/right arrow keys step one frame; space plays/pauses.</div>
      </section>
      <section class="panel" aria-label="Round-trip consistency plots">
        <h2>Round-trip consistency C_i over the episode</h2>
        <div class="controls">
          <div class="group" id="metric-toggles"><span class="label">Metric</span></div>
          <div class="group" id="depth-toggles"><span class="label">Depth</span></div>
          <div class="group">
            <label class="toggle"><input type="checkbox" id="y-zero" checked> y from 0</label>
            <label class="toggle"><input type="checkbox" id="x-full"> full step budget</label>
          </div>
        </div>
        <div id="charts"></div>
        <table class="readout" id="readout" aria-label="C_i at the current frame"></table>
      </section>
    </div>
  </main>
</div>
<script type="application/json" id="viewer-data">__DATA__</script>
<script>"""
    + _THEME_SCRIPT
    + """</script>
<script>__SCRIPT__</script>
</body>
</html>
"""
)

_TASK_SCRIPT = r"""
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("viewer-data").textContent);
  const EPISODES = DATA.episodes;
  const METRICS = [
    { key: "action_c_i", label: "C_i (action)" },
    { key: "latent_c_i", label: "C_i (latent)" },
  ];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const W = 640, H = 220, M = { top: 12, right: 14, bottom: 30, left: 56 };

  const depths = Array.from(new Set(EPISODES.flatMap((e) => e.depths))).sort((a, b) => a - b);
  const depthColor = {};
  depths.forEach((d, i) => { depthColor[d] = `var(--series-${(i % 6) + 1})`; });
  const state = {
    episode: null, frame: 0, filter: "all",
    metrics: new Set(METRICS.map((m) => m.key)), depths: new Set(depths),
    yZero: true, xFull: false,
  };

  const video = document.getElementById("video");
  const list = document.getElementById("episode-list");
  const chartsEl = document.getElementById("charts");
  const readout = document.getElementById("readout");

  // ---------- helpers ----------
  const el = (tag, attrs, text) => {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const svg = (tag, attrs) => {
    const n = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const fmt = (v) => {
    if (v === null || v === undefined) return "-";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(2);
    return v.toFixed(a < 1 ? 4 : 3);
  };
  // Axis labels share one precision, set by the tick spacing, so a column of
  // ticks reads consistently (0.0, 0.5, 1.0 - not 0.0000 next to 1.000).
  const tickFmt = (v, step) => {
    if (Math.abs(v) >= 1e4) return v.toExponential(1);
    const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(step) - 1e-9)));
    return v.toFixed(decimals);
  };
  const niceTicks = (lo, hi, count) => {
    if (!(hi > lo)) hi = lo + 1;
    const raw = (hi - lo) / count;
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw);
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(+t.toPrecision(12));
    return ticks;
  };
  const statusBadge = (ok) => {
    const wrap = el("span", { class: "status" });
    const icon = svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12", "aria-hidden": "true" });
    if (ok) {
      icon.appendChild(svg("path", { d: "M2 6.5 L5 9.2 L10 3", fill: "none", stroke: "var(--good)", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    } else {
      icon.appendChild(svg("path", { d: "M3 3 L9 9 M9 3 L3 9", fill: "none", stroke: "var(--critical)", "stroke-width": 2, "stroke-linecap": "round" }));
    }
    wrap.appendChild(icon);
    wrap.appendChild(document.createTextNode(ok ? "success" : "failure"));
    return wrap;
  };
  const meanAction = (ep) => {
    let s = 0, n = 0;
    for (const col of Object.values(ep.action_c_i)) for (const v of col) if (v !== null) { s += v; n++; }
    return n ? s / n : null;
  };

  // ---------- header ----------
  const nSucc = EPISODES.filter((e) => e.success).length;
  document.getElementById("task-meta").textContent =
    `${nSucc}/${EPISODES.length} succeeded (${(100 * nSucc / EPISODES.length).toFixed(1)}%)`;

  // ---------- episode list ----------
  function renderList() {
    list.replaceChildren();
    EPISODES.forEach((ep, i) => {
      if (state.filter === "success" && !ep.success) return;
      if (state.filter === "failure" && ep.success) return;
      const li = el("li");
      const b = el("button", { type: "button", "aria-current": String(state.episode === ep), title: ep.video });
      b.appendChild(el("span", { class: "ep-index" }, `#${i + 1}`));
      b.appendChild(statusBadge(ep.success));
      const m = meanAction(ep);
      b.appendChild(el("span", { class: "ep-ci", title: "mean action C_i (all depths)" }, m === null ? "" : `C̄ ${fmt(m)}`));
      b.addEventListener("click", () => selectEpisode(ep, i));
      li.appendChild(b);
      list.appendChild(li);
    });
  }
  document.querySelectorAll(".filters button").forEach((b) => {
    b.addEventListener("click", () => {
      state.filter = b.dataset.filter;
      document.querySelectorAll(".filters button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderList();
    });
  });

  // ---------- toggles ----------
  const metricToggles = document.getElementById("metric-toggles");
  METRICS.forEach((m) => {
    const lab = el("label", { class: "toggle" });
    const cb = el("input", { type: "checkbox" });
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) state.metrics.add(m.key); else state.metrics.delete(m.key);
      renderCharts();
    });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(m.label));
    metricToggles.appendChild(lab);
  });
  const depthToggles = document.getElementById("depth-toggles");
  depths.forEach((d) => {
    const lab = el("label", { class: "toggle" });
    const cb = el("input", { type: "checkbox" });
    cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) state.depths.add(d); else state.depths.delete(d);
      renderCharts();
    });
    const key = el("span", { class: "key" });
    key.style.background = depthColor[d];
    lab.appendChild(cb);
    lab.appendChild(key);
    lab.appendChild(document.createTextNode(`depth ${d}`));
    depthToggles.appendChild(lab);
  });
  document.getElementById("y-zero").addEventListener("change", (e) => { state.yZero = e.target.checked; renderCharts(); });
  document.getElementById("x-full").addEventListener("change", (e) => { state.xFull = e.target.checked; renderCharts(); });

  // ---------- charts ----------
  let charts = [];
  function renderCharts() {
    chartsEl.replaceChildren();
    charts = [];
    const ep = state.episode;
    if (!ep) return;
    const steps = ep.frame_steps;
    const xMax = state.xFull ? ep.max_episode_steps : Math.max(steps[steps.length - 1] || 1, 1);
    const visibleDepths = depths.filter((d) => state.depths.has(d) && ep.depths.includes(d));
    for (const m of METRICS) {
      if (!state.metrics.has(m.key)) continue;
      let lo = Infinity, hi = -Infinity;
      for (const d of visibleDepths) for (const v of ep[m.key][String(d)]) if (v !== null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (state.yZero) lo = Math.min(0, lo);
      if (hi === lo) hi = lo + (Math.abs(lo) || 1);
      const pad = (hi - lo) * 0.08;
      const yTicks = niceTicks(lo, hi + pad, 4);
      const yLo = Math.min(lo, yTicks[0]), yHi = Math.max(hi + pad, yTicks[yTicks.length - 1]);
      const xs = (s) => M.left + (s / xMax) * (W - M.left - M.right);
      const ys = (v) => H - M.bottom - ((v - yLo) / (yHi - yLo)) * (H - M.top - M.bottom);

      const wrap = el("div", { class: "chart" });
      wrap.appendChild(el("div", { class: "chart-title" }, m.label));
      const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${m.label} versus step` });
      for (const t of yTicks) {
        const y = ys(t);
        s.appendChild(svg("line", { x1: M.left, x2: W - M.right, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1 }));
        const lbl = svg("text", { x: M.left - 6, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "var(--text-muted)" });
        lbl.textContent = tickFmt(t, yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1);
        s.appendChild(lbl);
      }
      for (const t of niceTicks(0, xMax, 6)) {
        if (t > xMax) continue;
        const x = xs(t);
        const lbl = svg("text", { x, y: H - M.bottom + 16, "text-anchor": "middle", "font-size": 11, fill: "var(--text-muted)" });
        lbl.textContent = String(t);
        s.appendChild(lbl);
      }
      s.appendChild(svg("line", { x1: M.left, x2: W - M.right, y1: H - M.bottom, y2: H - M.bottom, stroke: "var(--axis)", "stroke-width": 1 }));
      const xl = svg("text", { x: W - M.right, y: H - 2, "text-anchor": "end", "font-size": 11, fill: "var(--text-muted)" });
      xl.textContent = "step";
      s.appendChild(xl);
      for (const d of visibleDepths) {
        const col = ep[m.key][String(d)];
        let path = "", open = false;
        for (let k = 0; k < col.length; k++) {
          const v = col[k];
          if (v === null) { open = false; continue; }
          path += `${open ? "L" : "M"}${xs(steps[k]).toFixed(1)},${ys(v).toFixed(1)}`;
          open = true;
        }
        if (path) s.appendChild(svg("path", { d: path, fill: "none", stroke: depthColor[d], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      }
      const hover = svg("line", { y1: M.top, y2: H - M.bottom, stroke: "var(--hover)", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" });
      const cursor = svg("line", { y1: M.top, y2: H - M.bottom, stroke: "var(--cursor)", "stroke-width": 1.5 });
      s.appendChild(hover);
      s.appendChild(cursor);
      const tip = el("div", { class: "tooltip" });
      wrap.appendChild(s);
      wrap.appendChild(tip);
      chartsEl.appendChild(wrap);
      const chart = { metric: m, svg: s, xs, cursor, hover, tip, visibleDepths, wrap };
      charts.push(chart);
      attachPointer(chart);
    }
    if (!charts.length) chartsEl.appendChild(el("div", { class: "hint" }, "No metric selected."));
    updateCursor();
  }

  const frameAtX = (chart, clientX) => {
    const r = chart.svg.getBoundingClientRect();
    const x = ((clientX - r.left) / r.width) * W;
    const steps = state.episode.frame_steps;
    let best = 0, bestD = Infinity;
    for (let k = 0; k < steps.length; k++) {
      const d = Math.abs(chart.xs(steps[k]) - x);
      if (d < bestD) { bestD = d; best = k; }
    }
    return best;
  };

  function attachPointer(chart) {
    let dragging = false;
    chart.svg.addEventListener("pointermove", (e) => {
      const k = frameAtX(chart, e.clientX);
      showTooltip(chart, k, e);
      if (dragging) seekToFrame(k);
    });
    chart.svg.addEventListener("pointerleave", () => { chart.hover.setAttribute("visibility", "hidden"); chart.tip.style.display = "none"; });
    chart.svg.addEventListener("pointerdown", (e) => {
      dragging = true;
      chart.svg.setPointerCapture(e.pointerId);
      seekToFrame(frameAtX(chart, e.clientX));
    });
    chart.svg.addEventListener("pointerup", () => { dragging = false; });
  }

  function showTooltip(chart, k, e) {
    const ep = state.episode;
    const x = chart.xs(ep.frame_steps[k]);
    chart.hover.setAttribute("x1", x);
    chart.hover.setAttribute("x2", x);
    chart.hover.setAttribute("visibility", "visible");
    const tip = chart.tip;
    tip.replaceChildren();
    tip.appendChild(el("div", { class: "t-head" }, `step ${ep.frame_steps[k]}`));
    for (const d of chart.visibleDepths) {
      const row = el("div", { class: "t-row" });
      const key = el("span", { class: "key" });
      key.style.background = depthColor[d];
      row.appendChild(key);
      row.appendChild(el("strong", null, fmt(ep[chart.metric.key][String(d)][k])));
      row.appendChild(document.createTextNode(` depth ${d}`));
      tip.appendChild(row);
    }
    tip.style.display = "block";
    const r = chart.wrap.getBoundingClientRect();
    let left = e.clientX - r.left + 12;
    if (left + tip.offsetWidth > r.width) left = e.clientX - r.left - tip.offsetWidth - 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, e.clientY - r.top - tip.offsetHeight - 8)}px`;
  }

  // ---------- playback sync ----------
  const frameOfTime = (t) => {
    const ep = state.episode;
    return Math.max(0, Math.min(ep.frame_steps.length - 1, Math.floor(t * ep.fps + 1e-6)));
  };
  function seekToFrame(k) {
    const ep = state.episode;
    video.currentTime = (k + 0.5) / ep.fps;
    state.frame = k;
    updateCursor();
  }
  function updateCursor() {
    const ep = state.episode;
    if (!ep) return;
    const k = state.frame;
    const step = ep.frame_steps[k];
    for (const c of charts) {
      const x = c.xs(step);
      c.cursor.setAttribute("x1", x);
      c.cursor.setAttribute("x2", x);
    }
    document.getElementById("cur-step").textContent = `step ${step} / ${ep.max_episode_steps}`;
    document.getElementById("cur-frame").textContent = `frame ${k + 1} / ${ep.frame_steps.length}`;
    renderReadout(k);
  }
  function renderReadout(k) {
    const ep = state.episode;
    readout.replaceChildren();
    const head = el("tr");
    head.appendChild(el("th", null, "depth"));
    for (const m of METRICS) head.appendChild(el("th", null, m.label));
    const thead = el("thead");
    thead.appendChild(head);
    readout.appendChild(thead);
    const tbody = el("tbody");
    for (const d of ep.depths) {
      const tr = el("tr");
      const first = el("td");
      const key = el("span", { class: "key" });
      key.style.background = depthColor[d];
      key.style.marginRight = "6px";
      key.style.verticalAlign = "middle";
      first.appendChild(key);
      first.appendChild(document.createTextNode(`depth ${d}`));
      tr.appendChild(first);
      for (const m of METRICS) tr.appendChild(el("td", null, fmt(ep[m.key][String(d)][k])));
      tbody.appendChild(tr);
    }
    readout.appendChild(tbody);
  }
  let raf = null;
  const tick = () => {
    if (state.episode) {
      const k = frameOfTime(video.currentTime);
      if (k !== state.frame) { state.frame = k; updateCursor(); }
    }
    raf = video.paused ? null : requestAnimationFrame(tick);
  };
  video.addEventListener("play", () => { if (!raf) raf = requestAnimationFrame(tick); });
  ["seeked", "timeupdate", "pause"].forEach((ev) => video.addEventListener(ev, () => {
    if (!state.episode) return;
    const k = frameOfTime(video.currentTime);
    if (k !== state.frame) { state.frame = k; updateCursor(); }
  }));
  document.addEventListener("keydown", (e) => {
    if (!state.episode || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      video.pause();
      seekToFrame(Math.max(0, Math.min(state.episode.frame_steps.length - 1, state.frame + (e.key === "ArrowRight" ? 1 : -1))));
    } else if (e.key === " " && e.target === document.body) {
      e.preventDefault();
      if (video.paused) video.play(); else video.pause();
    }
  });

  function selectEpisode(ep, i) {
    state.episode = ep;
    state.frame = 0;
    video.src = encodeURIComponent(ep.video);
    document.getElementById("video-title").textContent = `Episode #${i + 1} - ${ep.video}`;
    const st = document.getElementById("ep-status");
    st.replaceChildren(statusBadge(ep.success));
    renderList();
    renderCharts();
  }

  renderList();
  if (EPISODES.length) selectEpisode(EPISODES[0], 0);
})();
"""

_ROOT_TEMPLATE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C_i viewer: __TITLE__</title>
<style>__STYLE__</style>
</head>
<body class="viz-root">
<header>
  <h1>__TITLE__</h1>
  <span class="meta" id="root-meta"></span>
  <span class="spacer"></span>
  <button type="button" id="theme-toggle">Theme: auto</button>
</header>
<div class="root-main">
  <section class="panel">
    <h2>Tasks - open one to play its rollouts next to their C_i plots</h2>
    <div class="table-wrap"><table class="tasks" id="tasks"></table></div>
    <div class="hint">Mean C_i is action-space round-trip error, averaged over every recorded frame and depth of each episode, then over episodes.</div>
  </section>
</div>
<script type="application/json" id="viewer-data">__DATA__</script>
<script>"""
    + _THEME_SCRIPT
    + """</script>
<script>__SCRIPT__</script>
</body>
</html>
"""
)

_ROOT_SCRIPT = r"""
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("viewer-data").textContent);
  const el = (tag, attrs, text) => {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const fmt = (v) => (v === null || v === undefined ? "-" : v.toFixed(4));
  const table = document.getElementById("tasks");
  const head = el("tr");
  for (const h of ["Task", "Episodes", "Succeeded", "Success rate", "Mean C_i (failed)", "Mean C_i (succeeded)"]) head.appendChild(el("th", null, h));
  const thead = el("thead");
  thead.appendChild(head);
  table.appendChild(thead);
  const tbody = el("tbody");
  let n = 0, s = 0;
  for (const t of DATA.tasks) {
    n += t.n; s += t.n_success;
    const tr = el("tr");
    const td = el("td");
    td.appendChild(el("a", { href: encodeURI(t.href) }, t.task));
    tr.appendChild(td);
    tr.appendChild(el("td", null, String(t.n)));
    tr.appendChild(el("td", null, String(t.n_success)));
    tr.appendChild(el("td", null, `${(100 * t.n_success / t.n).toFixed(1)}%`));
    tr.appendChild(el("td", null, fmt(t.mean_ci_fail)));
    tr.appendChild(el("td", null, fmt(t.mean_ci_succ)));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  document.getElementById("root-meta").textContent =
    `${DATA.tasks.length} tasks, ${s}/${n} episodes succeeded (${n ? (100 * s / n).toFixed(1) : "-"}%)`;
})();
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path", type=Path, help="video folder, or a root of task folders with --root"
    )
    parser.add_argument(
        "--root", action="store_true", help="build every task subfolder plus a root index"
    )
    parser.add_argument("--title", default=None, help="page title (defaults to the folder name)")
    args = parser.parse_args()
    out = (
        build_root_index(args.path, title=args.title)
        if args.root
        else build_task_viewer(args.path, title=args.title)
    )
    if out is None:
        raise SystemExit(f"No C_i sidecars ({SIDECAR_FORMAT}) found under {args.path}")
    print(f"C_i viewer written to: {out}")


if __name__ == "__main__":
    main()
