// App shell: WebSocket + REST wiring, history, eval bar, status bar.
// Holds no game logic; renders whatever the server broadcasts.

import { Board } from "./board.js";
import { TreeView } from "./tree.js";
import { EditMode } from "./edit.js";
import { ParamsPanel } from "./params.js";
import { sparklinePoints, nearestPly } from "./sparkline.js";

const $ = (id) => document.getElementById(id);

let state = null; // last server state message
let lastStats = null;
// PV arrows (§11.1): the position they were computed for, and — when the
// engine just played pv[0] — the continuation to show on the next position
let arrowsFen = null;
let pendingArrows = null;
let sparkPts = []; // sparkline points of the last render, for clicks (§11.2)
let autoplayTimer = null; // pending next autoplay search (§11.3)
// §11.3: reused-tree searches can converge in milliseconds, so a Stop click
// usually lands *after* the search already ended naturally — the stop_reason
// alone cannot pause the chain. Every user command holds it instead; only
// deliberately starting a search (Move!, autoreply) re-arms it.
let autoplayHold = false;

/** Every explicit user command pauses the autoplay chain (§11.3). */
function cancelAutoplay() {
  clearTimeout(autoplayTimer);
  autoplayTimer = null;
  autoplayHold = true;
}

/** REST helper; returns the response JSON, or null on error (message shown). */
async function api(path, body, method = "POST") {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail;
    setMessage(detail || `${path} failed (${res.status})`);
    return null;
  }
  return res.json();
}

const treeView = new TreeView($("tree"), {
  // click-to-explore (§4.2): the clicked node's moves become real game moves
  onNodeClick: (path) => {
    if (!confirm(`Play ${path.join(" ")} into the game?`)) return;
    treeView.clearStep();
    cancelAutoplay();
    api("/api/goto", { path });
  },
  // fen: the base position of the client's tree snapshot (§10.1)
  fetchFens: (paths, fen) => api("/api/tree/fens", { paths, fen }),
  fetchDetail: (path) => api("/api/tree/detail", { root_path: path, max_nodes: 2000 }),
  onCollapseChange: (on) => {
    $("tree-collapse").textContent = on ? "⊞" : "⊟";
    $("tree-collapse").classList.toggle("active", on);
  },
  onCompressChange: (on) => $("tree-compress").classList.toggle("active", on),
});

const board = new Board($("board"), {
  onMove: async (uci) => {
    treeView.clearStep();
    cancelAutoplay();
    const s = await api("/api/move", { uci });
    // the common human-vs-engine flow: answer automatically (toggleable);
    // in infinite mode (§10.3) the reply is a fresh analysis, not a move
    if (s && !s.outcome && $("autoreply").checked) {
      autoplayHold = false; // deliberate start re-arms the chain (§11.3)
      api("/api/search/start", { analyse: $("infinite").checked });
    }
  },
});

const editMode = new EditMode(board, {
  apply: (fen) => api("/api/position", { fen }),
  onExit: async () => renderState(await fetch("/api/state").then((r) => r.json())),
});

const paramsPanel = new ParamsPanel($("params-body"), {
  put: async (body) => {
    const config = await api("/api/config", body, "PUT");
    if (config) paramsPanel.render(config);
    return config;
  },
});

// ---- websocket ------------------------------------------------------------

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/events`);
  ws.onmessage = (msg) => handle(JSON.parse(msg.data));
  ws.onclose = () => {
    setMessage("connection lost — retrying…");
    setTimeout(connect, 2000);
  };
  ws.onopen = () => setMessage("");
}

function handle(event) {
  if (event.type === "state") renderState(event);
  else if (event.type === "stats") renderStats(event);
  else if (event.type === "tree") {
    treeView.setTree(event);
    // stepping (§10.4) grows the tree across many tiny searches — the
    // cumulative count lives in the tree, not in the per-search stats
    if (state && !state.searching) renderIdleFooter();
  } else if (event.type === "config") {
    paramsPanel.render(event);
    treeView.cPuct = event.limits?.c_puct ?? 1.5; // hover PUCT breakdown
  } else if (event.type === "search_end") {
    renderStats(event);
    // the engine plays pv[0]: keep the expected continuation as arrows on
    // the position that is about to arrive (§11.1)
    pendingArrows =
      event.played_move && event.pv[0] === event.played_move ? event.pv.slice(1, 4) : null;
    // autoplay (§11.3): chain the next search only when the chain is armed
    // and the ending was natural (the reason check covers stops issued by
    // other clients, which this tab never held)
    if (
      !autoplayHold &&
      $("autoplay").checked &&
      event.played_move &&
      event.stop_reason !== "interrupted"
    ) {
      clearTimeout(autoplayTimer);
      autoplayTimer = setTimeout(() => {
        autoplayTimer = null;
        if (!autoplayHold && $("autoplay").checked && state && !state.searching && !state.outcome)
          api("/api/search/start");
      }, 600);
    }
    setMessage(
      event.played_move
        ? `engine played ${event.played_move} (${event.stop_reason})`
        : `search ended (${event.stop_reason})`,
    );
  }
}

// ---- rendering --------------------------------------------------------------

function renderState(s) {
  const fenChanged = state?.fen !== s.fen;
  state = s;
  if (!editMode.active) {
    const humanTurn = !s.searching && !s.outcome;
    board.setState(s, humanTurn);
  }
  if (fenChanged && s.fen !== arrowsFen) {
    // §11.1: after the engine's move show the PV continuation, otherwise
    // (human move, takeback, new game, edit) the arrows are stale — clear
    board.setArrows(pendingArrows ?? []);
    arrowsFen = pendingArrows ? s.fen : null;
    pendingArrows = null;
  }

  const infinite = $("infinite").checked;
  $("go").textContent = s.searching ? "■ Stop" : infinite ? "▶ Analyse" : "▶ Move!";
  $("go").disabled = Boolean(s.outcome);
  // infinite mode (§10.3): Stop never plays; this button does
  $("play-best").hidden = !infinite;
  $("play-best").disabled = Boolean(s.outcome);
  $("step").disabled = s.searching || Boolean(s.outcome);
  $("turn").textContent = s.outcome
    ? `${s.outcome.result} — ${s.outcome.termination}`
    : `${s.turn === "w" ? "white" : "black"} to move${s.searching ? " (thinking…)" : ""}`;

  const list = $("history");
  list.replaceChildren();
  s.history.forEach((san, i) => {
    if (i % 2 === 0) {
      const num = document.createElement("span");
      num.className = "movenum";
      num.textContent = `${i / 2 + 1}.`;
      list.appendChild(num);
    }
    const span = document.createElement("span");
    span.className = "move";
    span.textContent = san;
    // takeback (§3.3): rewind the game to just after this move
    span.addEventListener("click", () => {
      treeView.clearStep();
      cancelAutoplay();
      api("/api/goto", { ply: i + 1 });
    });
    list.appendChild(span);
  });
  list.scrollTop = list.scrollHeight;

  // keep the last search's metrics readable when idle (§10.1)
  if (!s.searching) renderIdleFooter();
  renderSparkline();
}

/** Eval history over the game (§11.2), plus a live point while searching. */
function renderSparkline() {
  const canvas = $("sparkline");
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !state) return; // panel hidden (edit mode)
  const entries = state.eval_history ?? [];
  const live =
    state.searching && lastStats
      ? { ply: state.history.length, white_win_prob: lastStats.white_win_prob }
      : null;
  // minimum span so early-game points don't stretch across the full width
  const span = Math.max(12, state.history.length, entries.at(-1)?.ply ?? 0, live?.ply ?? 0);
  sparkPts = sparklinePoints(entries, span, w, h);

  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.strokeStyle = "#4a4641"; // 50% midline
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  ctx.strokeStyle = "#e8e6e3";
  ctx.fillStyle = "#e8e6e3";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  sparkPts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
  ctx.stroke();
  for (const p of sparkPts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
  if (live) {
    const [p] = sparklinePoints([live], span, w, h);
    ctx.strokeStyle = "#6d9f4e"; // hollow accent dot: still moving
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function renderIdleFooter() {
  const tree = treeView.raw;
  const cumulative = tree && tree.visits[0] > 0 ? ` · tree ${fmt(tree.visits[0])} sims` : "";
  $("status-live").textContent =
    (lastStats ? `idle · ${statsLine(lastStats)}` : "idle") + cumulative;
}

function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

function statsLine(st) {
  return (
    `sims ${fmt(st.simulations)} · nodes ${fmt(st.nodes)} · ` +
    `${fmt(st.sims_per_s)} sims/s · ${fmt(st.nodes_per_s)} nodes/s · ` +
    `${(st.elapsed_ms / 1000).toFixed(1)}s · best ${st.best_move || "—"}`
  );
}

function renderStats(st) {
  lastStats = st;
  const pct = Math.max(2, Math.min(98, st.white_win_prob * 100));
  $("eval-white").style.height = `${pct}%`;
  const cp = st.white_cp;
  $("eval-label").textContent = (cp >= 0 ? "+" : "") + (cp / 100).toFixed(2);
  $("pv").textContent = st.pv_san.length ? st.pv_san.join(" ") : "—";
  treeView.setPV(st.pv, st.pv_san);
  board.setArrows(st.pv.slice(0, 3)); // §11.1: live PV on the board
  arrowsFen = state?.fen ?? null;
  renderSparkline(); // live eval point (§11.2)
  $("status-live").textContent = statsLine(st);
}

function setMessage(text) {
  $("message").textContent = text;
}

// ---- controls ----------------------------------------------------------------

$("go").addEventListener("click", () => {
  treeView.clearStep();
  cancelAutoplay();
  if (state?.searching) api("/api/search/stop");
  else {
    autoplayHold = false; // Move! (re)arms the autoplay chain (§11.3)
    api("/api/search/start", { analyse: $("infinite").checked });
  }
});
$("play-best").addEventListener("click", () => {
  treeView.clearStep();
  cancelAutoplay();
  api("/api/play/best");
});
$("infinite").addEventListener("change", () => {
  if (state) renderState(state); // relabel the go button, show/hide Move!
});
$("step").addEventListener("click", () => {
  // §10.4: highlight what the step(s) changed — capture the baseline first
  cancelAutoplay();
  treeView.markStep();
  api("/api/search/step", { steps: Math.max(1, Number($("step-n").value) || 1) });
});
$("new").addEventListener("click", () => {
  treeView.clearStep();
  cancelAutoplay();
  api("/api/new");
});
$("autoplay").addEventListener("change", (e) => {
  if (!e.target.checked) cancelAutoplay();
});
$("sparkline").addEventListener("click", (e) => {
  // §11.2: click a point to take the game back to that ply
  if (!state || state.searching) return;
  const x = e.clientX - e.target.getBoundingClientRect().left;
  const ply = nearestPly(sparkPts, x);
  if (ply === null || ply >= state.history.length) return;
  treeView.clearStep();
  cancelAutoplay();
  api("/api/goto", { ply });
});
$("flip").addEventListener("click", () => board.flip());
$("edit").addEventListener("click", () => {
  cancelAutoplay();
  if (editMode.active) return editMode.exit();
  if (!state) return;
  if (state.searching) return setMessage("stop the search before editing");
  showTab("board");
  editMode.enter(state.fen);
});

// ---- tabs -------------------------------------------------------------------

function showTab(tab) {
  $("view-board").hidden = tab !== "board";
  $("view-tree").hidden = tab !== "tree";
  $("tab-board").classList.toggle("active", tab === "board");
  $("tab-tree").classList.toggle("active", tab === "tree");
  history.replaceState(null, "", tab === "tree" ? "#tree" : "#");
  if (tab === "tree") {
    // while hidden the canvas was 0×0: its buffer is empty and trees that
    // arrived over the socket could not be fitted — refresh against the real
    // size, and fetch a tree if none has arrived yet this session
    requestAnimationFrame(() => {
      treeView.refresh();
      if (!treeView.tree) fetch("/api/tree").then((r) => r.json()).then((t) => treeView.setTree(t));
    });
  }
}
$("tab-board").addEventListener("click", () => showTab("board"));
$("tab-tree").addEventListener("click", () => showTab("tree"));
$("tree-fit").addEventListener("click", () => treeView.focusRoot());
$("tree-zoom-in").addEventListener("click", () => treeView.zoomBy(1.6));
$("tree-zoom-out").addEventListener("click", () => treeView.zoomBy(1 / 1.6));
$("tree-collapse").addEventListener("click", () => treeView.setCollapsed(!treeView.collapsed));
$("tree-compress").addEventListener("click", () => treeView.setCompressed(!treeView.compressed));
$("tree-compress-k").addEventListener("change", (e) => treeView.setCompressK(e.target.valueAsNumber));
$("tree-hover").addEventListener("change", (e) => treeView.setHoverEnabled(e.target.checked));

window.treeView = treeView; // debug / end-to-end test hook

// render immediately from REST; the socket takes over for live updates
fetch("/api/state").then((r) => r.json()).then(renderState);
fetch("/api/config").then((r) => r.json()).then((c) => paramsPanel.render(c));
if (location.hash === "#tree") showTab("tree");
connect();
