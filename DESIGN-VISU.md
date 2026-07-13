# Design Document: Browser-Based Visualization

Status: **draft for review**. Companion to `DESIGN.md`. Replaces the CLI as the
primary frontend; the CLI stays as the minimal fallback and for scripted use.

Guiding rule (unchanged): educational project — readability and simplicity win
over the last percent. The same applies to the frontend stack: as few moving
parts as possible.

---

## 1. Goals

1. **Board view** — play against the engine in the browser: new game, position
   editor (drag pieces on/off the board), a *Move!* button that lets the engine
   move for whichever side is to move, live evaluation while it thinks.
2. **Tree view** — a live debug view into the search tree while it builds up:
   semantic zoom from whole-tree silhouette down to per-node boards and
   statistics, click-to-explore, nodes/s in a status bar, and an editable panel
   for every parameter that influences the search.

Non-goals (for now): multiple concurrent games/clients, persistence/game
database, online play, mobile layout, replacing the CLI.

---

## 2. Technology Choice

The draft asked for a recommendation ("html5, but I'm not an expert"). Proposal:

- **Backend: FastAPI + uvicorn** in `python/chessengine/ui/web/`. It serves the
  static frontend, a small REST API for commands, and WebSockets for the live
  streams. It owns one `Game` + one `Engine` (a "session") and is the only
  writer to both — exactly the role `cli.py` has today. The dependency rule
  `ui → game → engine` is untouched; the C++ side needs one addition (§5).
- **Frontend: plain ES modules, no build step, no framework, no npm.** Static
  files served as-is; `uv sync && uv run chessengine-web` stays the only
  workflow. JSDoc type annotations where they help. Rationale: the frontend is
  two views and a settings panel — a bundler/framework buys little and costs a
  second toolchain. *Fallback:* if the tree renderer grows painful in plain JS,
  switch to Vite + TypeScript later; the no-build code ports 1:1.
- **Board rendering: own SVG component** (~300 lines: 64 squares, unicode/SVG
  pieces, pointer-event drag & drop). No external board library. Rationale:
  edit mode (spare-piece palette, drag-off-to-remove), PV arrows, and eval
  overlays all want custom control anyway; the popular off-the-shelf board
  (lichess's chessground) is GPL and brings a build step. The client contains
  **no chess rules** — legal target squares come with the server state, moves
  are validated server-side by `Game`.
- **Tree rendering: HTML5 `<canvas>` (2D), not SVG/DOM.** The tree reaches
  10^5–10^6 nodes; one DOM element per node is infeasible, canvas redraws the
  visible scene per frame. Pan/zoom via `d3-zoom` semantics but hand-rolled
  (one wheel + drag handler, a 2D transform) — no d3 dependency needed.

## 3. Board View

One page, two tabs (Board / Tree) sharing a bottom status bar.

### 3.1 Play mode

- Board with drag & drop and click-click moves; legal targets highlighted
  (from `state.legal_moves`). Promotion via a small popup. Last move and
  check highlighted. Flip button.
- **No fixed human color.** The board is always interactive for the side to
  move; **Move!** hands the current decision to the engine instead. This
  covers all of: human vs engine (either color), "switch sides mid-game", and
  engine vs engine (press Move! repeatedly; a small *auto* toggle keeps
  pressing it).
- While the engine thinks: eval bar beside the board (win prob + centipawns),
  sims/nodes/elapsed readout, PV shown as arrows on the board (first 2–3 PV
  plies) and as SAN text. A *Stop* button interrupts (`engine.stop()`; the
  best-so-far move is played).
- Move history (SAN, clickable → see takeback in §3.3), New game button.

### 3.2 Edit mode

- Toggle switches the board to free placement: spare-piece palettes (white /
  black) beside the board, drag onto the board to add, drag off to remove,
  *Clear* and *Start position* shortcuts. Side-to-move toggle; castling rights
  checkboxes (auto-derived from king/rook squares by default); en passant left
  out (edge case, FEN paste covers it). A FEN text field allows paste/copy.
- Validity (kings present, no pawns on back ranks, side not to move not in
  check, …) is checked server-side via `python-chess` `Board.is_valid()`;
  invalid positions can be built transiently but not *applied*.
- Leaving edit mode applies the position: new `Game(fen)`,
  `engine.set_position(fen)` — the search tree is dropped (by design).

### 3.3 Takeback / history navigation

Clicking a history entry rewinds the game to that point. The engine has no
"undo", so this is `set_position(fen)` — the tree is dropped. Acceptable: tree
reuse (`advance`) only pays going *forward*, which the normal flow preserves.

## 4. Tree View

A live, explorable rendering of the search tree, updating while (and after)
the engine thinks.

### 4.1 Semantic zoom (level of detail)

What is drawn per node depends on its on-screen size:

| LOD | node size on screen | rendering |
|---|---|---|
| L0 far | < ~3 px | edges only — tree silhouette to judge breadth/depth; edge alpha/width ∝ visits |
| L1 mid | ~3–24 px | disc, **fill = side to move at that node** (white/black), radius ∝ log(visits); edge width ∝ visits |
| L2 near | ~24–120 px | stat card: move (SAN), N (visits), win frequency W/N, score in cp, prior P |
| L3 max | > ~120 px | L2 + rendered mini board of the node's position |

- Layout: classic tidy/layered tree (root left, depth → x), children ordered by
  visit count. Recomputed client-side per snapshot; nodes keyed by **move path**
  (§5.2) so positions stay stable across updates and re-layouts animate gently.
- Hover: tooltip with full stats + UCT term breakdown (Q, exploration term) —
  cheap and very useful for understanding selection.
- The **current PV is highlighted** as a colored path from the root.

### 4.2 Live exploration ("click a node to continue from there")

Clicking a node **plays the moves along its path into the real game**: each
move goes through `game.push()` + `engine.advance()` — so the clicked subtree
is *kept* (tree reuse) and becomes the new root; the search continues/restarts
from there. The board tab reflects the new position and history.

This is deliberately "the exploration *is* the game" — simple, no shadow
state. Going back is a takeback (§3.3, tree dropped). A confirmation is shown
when the click would discard >1 move of real game history.

### 4.3 Status bar & parameter panel

- Status bar (both tabs): state (idle/searching/stop reason), simulations,
  nodes, **nodes/s and sims/s** (server-computed from successive `stats()`
  deltas), elapsed, eval, best move.
- Parameter panel (drawer in the tree view): every knob that influences the
  search, two groups with different lifecycles:
  - **Search limits** (`max_time_ms`, `max_simulations`, convergence on/off,
    `convergence_window`, `convergence_cp_threshold`, `c_puct`,
    `virtual_loss`): applied at the **next search start**. If a search is
    running, changes queue up and are marked "pending".
  - **Structural** (`workers`, `batch_size`, `seed`): require engine
    reconstruction. Applying stops the search, rebuilds `Engine`,
    `set_position(current fen)` — **tree is dropped**; the UI says so.

  (`c_puct`/`virtual_loss` are constructor config today; they move to
  `SearchLimits`-style per-search parameters in C++ so they land in the first
  group — trivial change, they're read per-selection anyway.)

## 5. Data Flow: Streaming the Tree

The hard problem: the tree can hit millions of nodes; shipping it whole to the
browser every second is out. Two streams with different budgets:

### 5.1 Stats stream (cheap, frequent)

WebSocket `/ws/events`, ~4 Hz while searching: `SearchStats` + derived rates +
game-state changes (fen/history after any move). Powers eval bar, status bar,
PV arrows. This is exactly the existing `stats()` poll from the CLI, relocated.

### 5.2 Tree stream (bounded, ~1 Hz)

Same WebSocket, ~1 Hz while searching (and once on search end / on demand):
a **filtered snapshot** of the most relevant part of the tree.

- New engine call `tree_view(max_nodes, min_visits, root_path=[])`:
  best-first walk from `root_path` (default: search root) taking the
  **`max_nodes` most-visited nodes** (default ~20 000 — enough that L0/L1 look
  like "the whole tree"), returning flat parallel arrays:
  `parent_index (int32)`, `move (uci)`, `visits (uint32)`, `q (float32)`,
  `prior (float32)`, `num_children_total (uint16)` (so the UI can show "…37
  children pruned"). Sent as JSON first; switch to binary only if profiling
  says so.
- **Node identity = move path from the root** (the UCI sequence). Arena
  indices are not stable across `advance()`; paths are, and they double as the
  payload for click-to-explore and board-thumbnail requests. The client keys
  its layout by path hash.
- **Zooming into pruned regions:** when the viewport centers on a node whose
  children were cut off, the client requests `tree_view(root_path=that path)`
  for the subtree — same call, deeper detail. Detail-on-demand instead of
  ever growing the global snapshot.
- **L3 board thumbnails** need positions only for the handful of visible
  nodes: client sends the node paths, the **server** replays them in
  python-chess and returns FENs, the client renders mini boards from FEN with
  the same SVG board component. No C++ involvement, no FENs in bulk snapshots.

### 5.3 C++ addition: concurrent `tree_view`

`tree_snapshot()` (M5) runs after search; `tree_view` must run **while workers
write**. This is safe with the existing memory discipline (DESIGN.md §4.1):
arena chunks never move, and children are only traversed when
`expand_state == EXPANDED` (acquire load) — the same protocol selection uses.
Visit/value reads are racy-but-monotonic like `stats()`: numbers may be a few
simulations stale or mutually inconsistent; for a debug view that's fine and
worth stating in the code. Runs on the caller's (Python) thread with the GIL
released; cost is a bounded best-first walk (priority queue over ≤ max_nodes
candidates), well under 10 ms for 20k nodes.

## 6. Server API

`python/chessengine/ui/web/`:

```
server.py        # FastAPI app; owns Game + Engine, the single writer
static/          # index.html, app.js, board.js, tree.js, …  (no build step)
```

Entry point `uv run chessengine-web` (console script) → starts uvicorn on
localhost, prints/opens the URL. One session per process; a second browser tab
sees the same game (last writer wins — documented, not solved).

REST (commands; all return the new state):

```
GET  /api/state                  # fen, turn, legal_moves, history, outcome,
                                 # searching?, edit_mode?
POST /api/move        {uci}      # human move: game.push + engine.advance
POST /api/new         {fen?}     # new game (optionally from FEN)
POST /api/position    {fen}      # apply edited position (validated)
POST /api/search/start {limits?} # engine.start; "Move!" = start + auto-play on stop
POST /api/search/stop
POST /api/goto        {ply | path}  # takeback / click-to-explore (§4.2)
GET  /api/config                 # current EngineConfig + SearchLimits + pending
PUT  /api/config      {...}      # split into limits vs structural (§4.3)
POST /api/tree/detail {root_path, max_nodes}   # subtree on demand (§5.2)
POST /api/tree/fens   {paths}    # FENs for visible L3 nodes
```

WebSocket `/ws/events` (server → client push):

```
{type: "stats",  ...SearchStats, sims_per_s, nodes_per_s}
{type: "tree",   root_path, nodes: {parents, moves, visits, q, prior, kids_total}}
{type: "state",  ...}            # after any game-state change
{type: "search_end", stop_reason, played_move?}
```

The centipawn↔win-prob mapping constant (DESIGN.md §8) is served in
`/api/config` so the JS eval bar uses the same curve as the engine.

## 7. Milestones

- **V1 — board view:** FastAPI server + SVG board; play vs engine with live
  stats + eval bar (CLI feature parity in the browser). Move!, New, Stop.
- **V2 — tree view, read-only:** `tree_view()` in C++, snapshot stream,
  canvas renderer with L0–L2, pan/zoom, PV highlight, status bar with nodes/s.
- **V3 — interaction:** L3 board thumbnails, subtree detail-on-demand,
  click-to-explore, parameter panel (both lifecycles), edit mode.
- **V4 — polish:** PV arrows on the main board, eval history sparkline,
  auto-play toggle, hover UCT breakdown.

Each milestone is shippable and demo-able on its own.

## 8. Testing

| Layer | Test |
|---|---|
| `tree_view` (C++) | filtering/ordering unit tests; **TSan gate: snapshot in a loop while a parallel search runs** |
| Server | FastAPI TestClient: move/new/goto/config round-trips; WS event sequences with a fake engine |
| Frontend | keep logic in pure functions (layout, LOD choice, path hashing) — unit-testable with node's built-in test runner, no browser harness for now |
| Manual | `verify` skill / checklist per milestone (drag, edit, explore) |

## 9. Open Questions (input wanted)

1. **Piece graphics:** unicode glyphs (zero assets, font-dependent rendering)
   vs vendoring one public-domain SVG piece set (crisper, ~15 small files).
   Recommendation: vendor SVG pieces (they're also needed for L3 thumbnails).
2. **Tree layout direction:** left→right layered (reads like a game, depth =
   x, good for wide trees) vs top-down. Recommendation: left→right.
3. **Move! semantics on game end / while searching** — button disabled, or
   Stop-then-move? Recommendation: Move! during a search acts as Stop (play
   best-so-far).
4. Keep the CLI at feature parity, or freeze it at M5 level and let the web
   UI pull ahead? Recommendation: freeze (it remains the smoke-test frontend).
