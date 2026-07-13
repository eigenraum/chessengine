// App shell: WebSocket + REST wiring, history, eval bar, status bar.
// Holds no game logic; renders whatever the server broadcasts.

import { Board } from "./board.js";
import { TreeView } from "./tree.js";

const $ = (id) => document.getElementById(id);

let state = null; // last server state message
let lastStats = null;

const treeView = new TreeView($("tree"));

const board = new Board($("board"), {
  onMove: async (uci) => {
    const s = await api("/api/move", { uci });
    // the common human-vs-engine flow: answer automatically (toggleable)
    if (s && !s.outcome && $("autoreply").checked) api("/api/search/start");
  },
});

/** POST helper; returns the new state, or null on error (message shown). */
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
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
  else if (event.type === "tree") treeView.setTree(event);
  else if (event.type === "search_end") {
    renderStats(event);
    setMessage(
      event.played_move
        ? `engine played ${event.played_move} (${event.stop_reason})`
        : `search ended (${event.stop_reason})`,
    );
  }
}

// ---- rendering --------------------------------------------------------------

function renderState(s) {
  state = s;
  const humanTurn = !s.searching && !s.outcome;
  board.setState(s, humanTurn);

  $("go").textContent = s.searching ? "■ Stop" : "▶ Move!";
  $("go").disabled = Boolean(s.outcome);
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
    list.appendChild(span);
  });
  list.scrollTop = list.scrollHeight;

  if (!s.searching) $("status-live").textContent = "idle";
}

function fmt(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

function renderStats(st) {
  lastStats = st;
  const pct = Math.max(2, Math.min(98, st.white_win_prob * 100));
  $("eval-white").style.height = `${pct}%`;
  const cp = st.white_cp;
  $("eval-label").textContent = (cp >= 0 ? "+" : "") + (cp / 100).toFixed(2);
  $("pv").textContent = st.pv_san.length ? st.pv_san.join(" ") : "—";
  treeView.setPV(st.pv);
  $("status-live").textContent =
    `sims ${fmt(st.simulations)} · nodes ${fmt(st.nodes)} · ` +
    `${fmt(st.sims_per_s)} sims/s · ${fmt(st.nodes_per_s)} nodes/s · ` +
    `${(st.elapsed_ms / 1000).toFixed(1)}s · best ${st.best_move || "—"}`;
}

function setMessage(text) {
  $("message").textContent = text;
}

// ---- controls ----------------------------------------------------------------

$("go").addEventListener("click", () =>
  api(state?.searching ? "/api/search/stop" : "/api/search/start"),
);
$("new").addEventListener("click", () => api("/api/new"));
$("flip").addEventListener("click", () => board.flip());

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

// render immediately from REST; the socket takes over for live updates
fetch("/api/state").then((r) => r.json()).then(renderState);
if (location.hash === "#tree") showTab("tree");
connect();
