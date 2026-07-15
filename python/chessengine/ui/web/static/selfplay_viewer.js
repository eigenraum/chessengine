// Self-play shard viewer (docs/readme-training.md): loads an .npz written
// by chessengine-selfplay and steps through its reconstructed game. Reuses
// the Board class in read-only mode (interactive=false) — this is a replay
// of already-recorded data, not the live Game/Engine session, and is kept
// entirely client-side once loaded (no server-side "current ply" state).

const $ = (id) => document.getElementById(id);

export class SelfPlayViewer {
  /**
   * @param {import("./board.js").Board} board read-only board to render into
   * @param {{load: (path: string) => Promise<object|null>}} callbacks
   */
  constructor(board, { load }) {
    this.board = board;
    this.load = load;
    this.game = null; // {meta, positions}
    this.ply = 0;

    $("selfplay-load").addEventListener("click", () => this._load());
    $("selfplay-path").addEventListener("keydown", (e) => {
      if (e.key === "Enter") this._load();
    });
  }

  async _load() {
    const path = $("selfplay-path").value.trim();
    if (!path) return;
    $("selfplay-status").textContent = "loading…";
    const game = await this.load(path);
    if (!game) {
      $("selfplay-status").textContent = "";
      return;
    }
    this.game = game;
    this.ply = 0;
    const net = game.meta?.net ? ` — net ${game.meta.net}` : "";
    $("selfplay-status").textContent = `${game.positions.length} positions${net}`;
    this._renderHistory();
    this._renderPly();
  }

  _renderHistory() {
    const list = $("selfplay-history");
    list.replaceChildren();
    if (!this.game) return;
    this.game.positions.forEach((pos, i) => {
      if (pos.san == null) return; // the final, undecided position has no move
      if (i % 2 === 0) {
        const num = document.createElement("span");
        num.className = "movenum";
        num.textContent = `${i / 2 + 1}.`;
        list.appendChild(num);
      }
      const span = document.createElement("span");
      span.className = "move";
      span.dataset.ply = String(i + 1);
      span.textContent = pos.san;
      span.addEventListener("click", () => {
        this.ply = i + 1;
        this._renderPly();
      });
      list.appendChild(span);
    });
  }

  _renderPly() {
    if (!this.game) return;
    const pos = this.game.positions[Math.min(this.ply, this.game.positions.length - 1)];
    this.board.setState(
      { fen: pos.fen, legal_moves: [], last_move: null, check_square: null },
      false,
    );

    for (const el of $("selfplay-history").querySelectorAll(".move")) {
      el.classList.toggle("current", Number(el.dataset.ply) === this.ply);
    }

    $("selfplay-stats").replaceChildren(
      ..._statRows([
        ["value (side to move)", pos.search_value.toFixed(3)],
        ["visits", pos.visit_count.toLocaleString()],
        ["game outcome (side to move)", pos.outcome.toFixed(1)],
      ]),
    );
    $("selfplay-policy").replaceChildren(
      ..._statRows(pos.policy.slice(0, 8).map((p) => [p.move, `${(p.prob * 100).toFixed(1)}%`])),
    );
  }
}

function _statRows(pairs) {
  return pairs.map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "stat-row";
    const a = document.createElement("span");
    a.textContent = label;
    const b = document.createElement("span");
    b.textContent = value;
    row.append(a, b);
    return row;
  });
}
