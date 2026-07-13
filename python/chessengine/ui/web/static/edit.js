// Edit mode (DESIGN-VISU.md §3.2): free-placement board editing with spare
// piece palettes, side-to-move / castling controls and a FEN field. Purely
// client-side while editing; validity is the server's call (Board.is_valid)
// when the position is applied.

import { pieceElement } from "./pieces.js";
import { fenPlacement } from "./board.js";

const FILES = "abcdefgh";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/** placement Map + toggles -> FEN. En passant is left out by design (§3.2);
 * paste a full FEN into the field for that edge case. */
export function composeFen(placement, turn, castling) {
  const rows = [];
  for (let rank = 8; rank >= 1; rank--) {
    let row = "";
    let empty = 0;
    for (let file = 0; file < 8; file++) {
      const symbol = placement.get(FILES[file] + rank);
      if (symbol) {
        if (empty > 0) row += empty;
        empty = 0;
        row += symbol;
      } else {
        empty++;
      }
    }
    if (empty > 0) row += empty;
    rows.push(row);
  }
  return `${rows.join("/")} ${turn} ${castling || "-"} - 0 1`;
}

/** Which castling rights the piece placement permits (king/rook at home). */
export function castlingAvailable(placement) {
  return {
    K: placement.get("e1") === "K" && placement.get("h1") === "R",
    Q: placement.get("e1") === "K" && placement.get("a1") === "R",
    k: placement.get("e8") === "k" && placement.get("h8") === "r",
    q: placement.get("e8") === "k" && placement.get("a8") === "r",
  };
}

export class EditMode {
  /**
   * @param {import("./board.js").Board} board
   * @param {{apply: (fen: string) => Promise<object|null>, onExit: () => void}} callbacks
   */
  constructor(board, { apply, onExit }) {
    this.board = board;
    this.apply = apply;
    this.onExit = onExit;
    this.active = false;
    this.spare = null; // symbol being dragged from the palette

    this.$ = (id) => document.getElementById(id);
    this._buildPalette();
    board.onEdit = () => this._placementChanged();

    this.$("edit-start").addEventListener("click", () => {
      this.board.setPlacement(fenPlacement(START_FEN));
    });
    this.$("edit-clear").addEventListener("click", () => this.board.clearBoard());
    this.$("edit-apply").addEventListener("click", () => this._apply());
    this.$("edit-cancel").addEventListener("click", () => this.exit());
    this.$("edit-fen").addEventListener("change", () => this._fenTyped());
    for (const input of document.querySelectorAll("#edit-panel input[type=radio], .castle"))
      input.addEventListener("change", () => this._syncFen());

    document.addEventListener("pointerup", (e) => this._spareDrop(e));
  }

  enter(fen) {
    this.active = true;
    this.board.setState(
      { fen, legal_moves: [], last_move: null, check_square: null },
      false,
    );
    this.board.setEditable(true);
    this.$("turn-" + fen.split(" ")[1]).checked = true;
    this._placementChanged(fen.split(" ")[2]);
    document.body.classList.add("editing");
  }

  exit() {
    this.active = false;
    this.board.setEditable(false);
    document.body.classList.remove("editing");
    this.onExit();
  }

  async _apply() {
    const state = await this.apply(this.$("edit-fen").value.trim());
    if (state) this.exit(); // invalid position: server said why, stay in edit
  }

  // ---- palette -------------------------------------------------------------

  _buildPalette() {
    const palette = this.$("palette");
    for (const symbol of ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"]) {
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 45 45");
      svg.setAttribute("class", "spare");
      svg.appendChild(pieceElement(symbol));
      svg.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        this.spare = symbol;
        document.body.classList.add("spare-dragging");
      });
      palette.appendChild(svg);
    }
  }

  _spareDrop(event) {
    if (!this.spare) return;
    const symbol = this.spare;
    this.spare = null;
    document.body.classList.remove("spare-dragging");
    const square = this.board.squareAt(event);
    if (square) this.board.place(square, symbol);
  }

  // ---- FEN <-> controls ------------------------------------------------------

  /** After any placement change: refresh castling availability + FEN field. */
  _placementChanged(keepCastling = null) {
    const available = castlingAvailable(this.board.placement);
    for (const [right, ok] of Object.entries(available)) {
      const box = this.$("castle-" + (right === right.toUpperCase() ? "w" : "b") + right.toLowerCase());
      box.disabled = !ok;
      // default: take every right the placement permits (auto-derived, §3.2)
      box.checked = ok && (keepCastling === null || keepCastling.includes(right));
    }
    this._syncFen();
  }

  _syncFen() {
    const turn = this.$("turn-b").checked ? "b" : "w";
    let castling = "";
    for (const [right, id] of [["K", "castle-wk"], ["Q", "castle-wq"], ["k", "castle-bk"], ["q", "castle-bq"]])
      if (this.$(id).checked) castling += right;
    this.$("edit-fen").value = composeFen(this.board.placement, turn, castling);
  }

  _fenTyped() {
    const fen = this.$("edit-fen").value.trim();
    const parts = fen.split(/\s+/);
    if (parts.length < 1 || !parts[0].includes("/")) return;
    this.board.editable = false; // avoid onEdit clobbering the typed FEN
    this.board.setState({ fen, legal_moves: [], last_move: null, check_square: null }, false);
    this.board.setEditable(true);
    if (parts[1] === "w" || parts[1] === "b") this.$("turn-" + parts[1]).checked = true;
    this._placementChanged(parts[2] ?? "");
    this.$("edit-fen").value = fen; // keep exactly what was typed (ep, counters)
  }
}
