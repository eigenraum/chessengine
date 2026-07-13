// SVG chess board: rendering + move input (drag & drop and click-click).
// No chess rules live here — legal moves come from the server state, moves
// are submitted via the onMove callback (DESIGN-VISU.md §2).

import { pieceElement } from "./pieces.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const SQ = 45; // square size in viewBox units, matches the piece viewBox
const FILES = "abcdefgh";

function el(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

/** "e4" -> {file: 4, rank: 3} (0-based, rank 0 = white's first rank) */
function parseSquare(name) {
  return { file: FILES.indexOf(name[0]), rank: Number(name[1]) - 1 };
}

/** FEN board field -> Map("e4" -> "P") — placement parsing only, no rules. */
export function fenPlacement(fen) {
  const map = new Map();
  const rows = fen.split(" ")[0].split("/");
  rows.forEach((row, i) => {
    const rank = 7 - i;
    let file = 0;
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") file += Number(ch);
      else map.set(FILES[file++] + (rank + 1), ch);
    }
  });
  return map;
}

export class Board {
  /**
   * @param {SVGSVGElement} svg
   * @param {{onMove: (uci: string) => void}} callbacks
   */
  constructor(svg, { onMove }) {
    this.svg = svg;
    this.onMove = onMove;
    this.orientation = "w";
    this.interactive = false;
    this.legalMoves = []; // uci strings, from the server state
    this.placement = new Map();
    this.selected = null; // square name
    this.lastMove = null; // uci
    this.checkSquare = null;
    this.drag = null;

    svg.setAttribute("viewBox", `0 0 ${8 * SQ} ${8 * SQ}`);
    this.layers = {};
    for (const name of ["squares", "marks", "pieces", "overlay"]) {
      this.layers[name] = el("g");
      svg.appendChild(this.layers[name]);
    }
    this._drawSquares();

    svg.addEventListener("pointerdown", (e) => this._pointerDown(e));
    svg.addEventListener("pointermove", (e) => this._pointerMove(e));
    svg.addEventListener("pointerup", (e) => this._pointerUp(e));
  }

  /** Update from a server state message and redraw. */
  setState({ fen, legal_moves, last_move, check_square }, interactive) {
    this.placement = fenPlacement(fen);
    this.legalMoves = legal_moves;
    this.lastMove = last_move;
    this.checkSquare = check_square;
    this.interactive = interactive;
    this.selected = null;
    this.render();
  }

  flip() {
    this.orientation = this.orientation === "w" ? "b" : "w";
    this.render();
  }

  // ---- geometry ----------------------------------------------------------

  /** square name -> top-left viewBox coordinates, honoring orientation */
  _xy(square) {
    const { file, rank } = parseSquare(square);
    const x = this.orientation === "w" ? file : 7 - file;
    const y = this.orientation === "w" ? 7 - rank : rank;
    return { x: x * SQ, y: y * SQ };
  }

  /** pointer event -> square name (or null outside the board) */
  _square(event) {
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      this.svg.getScreenCTM().inverse(),
    );
    const fx = Math.floor(pt.x / SQ);
    const fy = Math.floor(pt.y / SQ);
    if (fx < 0 || fx > 7 || fy < 0 || fy > 7) return null;
    const file = this.orientation === "w" ? fx : 7 - fx;
    const rank = this.orientation === "w" ? 7 - fy : fy;
    return FILES[file] + (rank + 1);
  }

  // ---- rendering -----------------------------------------------------------

  _drawSquares() {
    for (let x = 0; x < 8; x++) {
      for (let y = 0; y < 8; y++) {
        this.layers.squares.appendChild(
          el("rect", {
            x: x * SQ,
            y: y * SQ,
            width: SQ,
            height: SQ,
            class: (x + y) % 2 === 0 ? "sq-light" : "sq-dark",
          }),
        );
      }
    }
  }

  render() {
    this._renderMarks();
    this._renderPieces();
  }

  _renderMarks() {
    const marks = this.layers.marks;
    marks.replaceChildren();
    if (this.lastMove) {
      for (const sq of [this.lastMove.slice(0, 2), this.lastMove.slice(2, 4)]) {
        const { x, y } = this._xy(sq);
        marks.appendChild(el("rect", { x, y, width: SQ, height: SQ, class: "mark-last" }));
      }
    }
    if (this.checkSquare) {
      const { x, y } = this._xy(this.checkSquare);
      marks.appendChild(
        el("circle", { cx: x + SQ / 2, cy: y + SQ / 2, r: SQ * 0.46, class: "mark-check" }),
      );
    }
    if (this.selected) {
      const { x, y } = this._xy(this.selected);
      marks.appendChild(el("rect", { x, y, width: SQ, height: SQ, class: "mark-selected" }));
      for (const target of this._targets(this.selected)) {
        const p = this._xy(target);
        const capture = this.placement.has(target);
        marks.appendChild(
          el("circle", {
            cx: p.x + SQ / 2,
            cy: p.y + SQ / 2,
            r: capture ? SQ * 0.44 : SQ * 0.14,
            class: capture ? "mark-capture" : "mark-target",
          }),
        );
      }
    }
  }

  _renderPieces() {
    this.layers.pieces.replaceChildren();
    for (const [square, symbol] of this.placement) {
      const g = pieceElement(symbol);
      const { x, y } = this._xy(square);
      g.setAttribute("transform", `translate(${x} ${y})`);
      g.dataset.square = square;
      this.layers.pieces.appendChild(g);
    }
  }

  // ---- move input ----------------------------------------------------------

  _targets(from) {
    return this.legalMoves.filter((m) => m.startsWith(from)).map((m) => m.slice(2, 4));
  }

  _pointerDown(event) {
    if (!this.interactive || this.drag) return;
    const square = this._square(event);
    if (!square) return;

    if (this.selected && this._targets(this.selected).includes(square)) {
      this._submit(this.selected, square);
      return;
    }
    if (this._targets(square).length > 0) {
      // pick up own piece: select it and start a drag
      this.selected = square;
      this.svg.setPointerCapture(event.pointerId);
      const node = this.layers.pieces.querySelector(`[data-square="${square}"]`);
      this.drag = { from: square, node, moved: false };
      node.classList.add("dragging");
      this._renderMarks();
    } else {
      this.selected = null;
      this._renderMarks();
    }
  }

  _pointerMove(event) {
    if (!this.drag) return;
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(
      this.svg.getScreenCTM().inverse(),
    );
    this.drag.moved = true;
    this.drag.node.setAttribute("transform", `translate(${pt.x - SQ / 2} ${pt.y - SQ / 2})`);
  }

  _pointerUp(event) {
    if (!this.drag) return;
    const { from, node, moved } = this.drag;
    this.drag = null;
    node.classList.remove("dragging");
    const target = this._square(event);
    if (moved && target && target !== from && this._targets(from).includes(target)) {
      this._submit(from, target);
    } else {
      // click (piece stays selected for click-click) or aborted drag: snap back
      const { x, y } = this._xy(from);
      node.setAttribute("transform", `translate(${x} ${y})`);
    }
  }

  _submit(from, to) {
    this.selected = null;
    const candidates = this.legalMoves.filter((m) => m.startsWith(from + to));
    if (candidates.length > 1) {
      // promotion: same from+to with q/r/b/n suffixes
      this._askPromotion(to, (piece) => this.onMove(from + to + piece));
    } else if (candidates.length === 1) {
      this.onMove(candidates[0]);
    }
    this._renderMarks();
  }

  _askPromotion(square, resolve) {
    const overlay = this.layers.overlay;
    overlay.replaceChildren();
    overlay.appendChild(
      el("rect", { x: 0, y: 0, width: 8 * SQ, height: 8 * SQ, class: "promo-backdrop" }),
    );
    const isWhite = square[1] === "8";
    const { x } = this._xy(square);
    ["q", "r", "b", "n"].forEach((piece, i) => {
      const y = this.orientation === "w" === isWhite ? i * SQ : (7 - i) * SQ;
      const cell = el("g", { transform: `translate(${x} ${y})`, class: "promo-choice" });
      cell.appendChild(el("rect", { width: SQ, height: SQ, class: "promo-cell" }));
      cell.appendChild(pieceElement(isWhite ? piece.toUpperCase() : piece));
      cell.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        overlay.replaceChildren();
        resolve(piece);
      });
      overlay.appendChild(cell);
    });
    overlay
      .querySelector(".promo-backdrop")
      .addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        overlay.replaceChildren();
      });
  }
}
