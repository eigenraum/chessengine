// Live search-tree view: canvas renderer with semantic zoom (DESIGN-VISU.md
// §4.1). Consumes the flat tree snapshots streamed by the server; draws
// edges only when far out (L0), side-to-move discs at mid zoom (L1), stat
// cards up close (L2) and cards with board thumbnails at max zoom (L3).
// No DOM per node — the tree has tens of thousands.
//
// Nodes are identified by their move path from the root (§5.2): that is the
// payload for click-to-explore, board-thumbnail requests, and grafting
// detail-on-demand subtrees into the snapshot.

import { PIECE_SVG } from "./pieces.js";
import { fenPlacement } from "./board.js";

const DX = 140; // world units per ply (depth → x)
const ROW = 16; // world units per leaf row (y)

// Sibling order, top-down. Pluggable criterion (a UI dropdown later);
// initially: visit count, most-visited child on top.
const SORT_CRITERIA = {
  visits: (tree) => (a, b) => tree.visits[b] - tree.visits[a],
};

/** win prob (mover's view) -> centipawns; same logistic as the engine. */
function toCp(p) {
  const clamped = Math.min(Math.max(p, 0.001), 0.999);
  return Math.round(-400 * Math.log10(1 / clamped - 1));
}

/** Layout: x = depth, leaves get consecutive rows, parents center on their
 * children. Returns {x, y, depth, children, rows}. Iterative DFS — PV lines
 * can be deep. */
export function layoutTree(tree, sortBy = "visits") {
  const n = tree.parent.length;
  const children = Array.from({ length: n }, () => []);
  const depth = new Int32Array(n);
  for (let i = 1; i < n; i++) {
    children[tree.parent[i]].push(i);
    depth[i] = depth[tree.parent[i]] + 1;
  }
  const compare = SORT_CRITERIA[sortBy](tree);
  for (const list of children) list.sort(compare);

  const x = new Float64Array(n);
  const y = new Float64Array(n);
  let rows = 0;
  let maxDepth = 0;
  if (n > 0) {
    // post-order: children first, then center the parent on them
    const stack = [[0, 0]]; // [node, next-child cursor]
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      const [node, cursor] = top;
      if (children[node].length === 0) {
        y[node] = rows++ * ROW;
        stack.pop();
      } else if (cursor < children[node].length) {
        top[1]++;
        stack.push([children[node][cursor], 0]);
      } else {
        const kids = children[node];
        y[node] = (y[kids[0]] + y[kids[kids.length - 1]]) / 2;
        stack.pop();
      }
      x[node] = depth[node] * DX;
      if (depth[node] > maxDepth) maxDepth = depth[node];
    }
  }
  return { x, y, depth, children, rows, maxX: maxDepth * DX };
}

// ---- piece images for canvas thumbnails -----------------------------------
// The SVG piece set rendered once per symbol into an <img> (data URI), then
// drawn with drawImage. Colors match the .piece CSS classes.

const pieceImages = new Map();

function pieceImage(symbol) {
  let img = pieceImages.get(symbol);
  if (img) return img;
  const white = symbol === symbol.toUpperCase();
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45">` +
    `<g fill="${white ? "#f6f6f6" : "#33302c"}" stroke="${white ? "#33302c" : "#dcd9d4"}"` +
    ` stroke-width="1.3" stroke-linejoin="round" stroke-linecap="round">` +
    `${PIECE_SVG[symbol.toLowerCase()]}</g></svg>`;
  img = new Image();
  img.src = "data:image/svg+xml," + encodeURIComponent(svg);
  pieceImages.set(symbol, img);
  return img;
}

export class TreeView {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{
   *   onNodeClick?: (path: string[]) => void,
   *   fetchFens?: (paths: string[][]) => Promise<{fens: (string|null)[], sans: (string|null)[]}|null>,
   *   fetchDetail?: (path: string[]) => Promise<object|null>,
   * }} callbacks
   */
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.callbacks = callbacks;
    this.tree = null;
    this.layout = null;
    this.pv = [];
    this.paths = null; // paths[i]: UCI move path from the game position
    this.pathIndex = null; // path key ("e2e4 e7e5") -> row
    this.fens = new Map(); // path key -> {fen, san} for L3 thumbnails
    this.fensPending = new Set();
    this.detailDone = new Set(); // subtrees already requested this snapshot
    // world -> css px: sx = x * kx + tx. Anisotropic on purpose: the tree is
    // a few dozen plies wide but tens of thousands of rows tall, so depth and
    // rows need independent scales; zooming moves both together.
    this.tf = { x: 40, y: 40, kx: 1, ky: 1 };
    this.userMoved = false; // stop auto-fitting once the user pans/zooms

    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("pointerdown", (e) => this._dragStart(e));
    canvas.addEventListener("pointermove", (e) => this._dragMove(e));
    canvas.addEventListener("pointerup", (e) => this._dragEnd(e));
    new ResizeObserver(() => this.refresh()).observe(canvas);
    this.refresh();
  }

  setTree(msg) {
    this.tree = msg;
    this.layout = layoutTree(msg);
    this.paths = null;
    this.pathIndex = null;
    this.fens.clear();
    this.fensPending.clear();
    this.detailDone.clear();
    if (!this.userMoved) this._fit();
    this.draw();
  }

  setPV(pv) {
    this.pv = pv;
    if (this.tree) this.draw();
  }

  // ---- node identity -------------------------------------------------------

  _ensurePaths() {
    if (this.paths) return;
    const tree = this.tree;
    const n = tree.parent.length;
    this.paths = new Array(n);
    this.pathIndex = new Map();
    this.paths[0] = tree.root_path || [];
    this.pathIndex.set(this.paths[0].join(" "), 0);
    for (let i = 1; i < n; i++) {
      const path = this.paths[tree.parent[i]].concat(tree.move[i]);
      this.paths[i] = path;
      this.pathIndex.set(path.join(" "), i);
    }
  }

  /** Merge a detail snapshot (tree_view at root_path) into the current tree:
   * known nodes get fresher stats, new ones are appended (§5.2 zooming into
   * pruned regions). Appending keeps parent[i] < i, so layout stays valid. */
  graftDetail(rootPath, msg) {
    if (!this.tree || !msg || msg.parent.length <= 1) return;
    this._ensurePaths();
    const anchor = this.pathIndex.get(rootPath.join(" "));
    if (anchor === undefined) return; // a newer base snapshot replaced the tree
    const tree = this.tree;
    const rowOf = new Array(msg.parent.length); // detail row -> tree row
    rowOf[0] = anchor;
    for (let i = 1; i < msg.parent.length; i++) {
      const parentRow = rowOf[msg.parent[i]];
      if (parentRow === undefined) continue;
      const path = this.paths[parentRow].concat(msg.move[i]);
      const key = path.join(" ");
      let row = this.pathIndex.get(key);
      if (row === undefined) {
        row = tree.parent.length;
        tree.parent.push(parentRow);
        tree.move.push(msg.move[i]);
        tree.visits.push(msg.visits[i]);
        tree.q.push(msg.q[i]);
        tree.prior.push(msg.prior[i]);
        tree.children_total.push(msg.children_total[i]);
        this.paths.push(path);
        this.pathIndex.set(key, row);
      } else {
        tree.visits[row] = msg.visits[i];
        tree.q[row] = msg.q[i];
        tree.children_total[row] = msg.children_total[i];
      }
      rowOf[i] = row;
    }
    this.layout = layoutTree(tree);
    this.draw();
  }

  // ---- viewport -----------------------------------------------------------

  /** Re-read the canvas size and redraw. On a hidden tab the canvas is 0×0
   * and trees arriving then can't be fitted, so when the tab (re)appears the
   * view must be fitted against the now-real size. */
  refresh() {
    const dpr = window.devicePixelRatio || 1;
    // Cap the buffer (huge = slow or silently unpaintable) and only touch
    // the attributes on change — assigning resets the canvas.
    const w = Math.min(Math.round(this.canvas.clientWidth * dpr), 8192);
    const h = Math.min(Math.round(this.canvas.clientHeight * dpr), 8192);
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in css px below
    if (this.tree && !this.userMoved) this._fit();
    this.draw();
  }

  _fit() {
    const { width, height } = this.canvas.getBoundingClientRect();
    if (width === 0 || this.layout.rows === 0) return;
    const worldW = this.layout.maxX + DX;
    const worldH = Math.max(this.layout.rows * ROW, ROW);
    this.tf.kx = Math.min((width - 80) / worldW, 1.3);
    this.tf.ky = Math.min((height - 80) / worldH, 1.5);
    this.tf.x = 40;
    this.tf.y = height / 2 - (worldH / 2) * this.tf.ky;
  }

  /** Refit the whole tree, root at the left edge, and return to auto-fit. */
  focusRoot() {
    this.userMoved = false;
    if (!this.tree) return;
    this._fit();
    this.draw();
  }

  /** Zoom by `factor` around the canvas center (the +/- buttons). */
  zoomBy(factor) {
    const { width, height } = this.canvas.getBoundingClientRect();
    this._zoomAt(width / 2, height / 2, factor);
  }

  _zoomAt(mx, my, factor) {
    // kx is capped: past card zoom, more zoom spreads rows, not plies
    const kx = Math.min(Math.max(this.tf.kx * factor, 0.002), 1.4);
    const ky = Math.min(Math.max(this.tf.ky * factor, 0.0005), 12);
    this.tf.x = mx - ((mx - this.tf.x) * kx) / this.tf.kx;
    this.tf.y = my - ((my - this.tf.y) * ky) / this.tf.ky;
    this.tf.kx = kx;
    this.tf.ky = ky;
    this.userMoved = true;
    this.draw();
  }

  _wheel(event) {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    this._zoomAt(
      event.clientX - rect.left,
      event.clientY - rect.top,
      Math.exp(-event.deltaY * 0.0015),
    );
  }

  _dragStart(event) {
    this.dragFrom = { x: event.clientX, y: event.clientY };
    this.dragDist = 0;
    this.canvas.setPointerCapture(event.pointerId);
  }

  _dragMove(event) {
    if (!this.dragFrom) return;
    const dx = event.clientX - this.dragFrom.x;
    const dy = event.clientY - this.dragFrom.y;
    this.dragDist += Math.abs(dx) + Math.abs(dy);
    if (this.dragDist > 4) this.userMoved = true;
    this.tf.x += dx;
    this.tf.y += dy;
    this.dragFrom = { x: event.clientX, y: event.clientY };
    this.draw();
  }

  _dragEnd(event) {
    const wasClick = this.dragFrom && this.dragDist <= 4;
    this.dragFrom = null;
    if (wasClick) this._click(event);
  }

  // ---- click-to-explore (§4.2) ---------------------------------------------

  _click(event) {
    if (!this.tree || !this.callbacks.onNodeClick) return;
    const rect = this.canvas.getBoundingClientRect();
    const hit = this._hitTest(event.clientX - rect.left, event.clientY - rect.top);
    if (hit === null || hit === 0) return; // root = current position, no-op
    this._ensurePaths();
    this.callbacks.onNodeClick(this.paths[hit]);
  }

  /** Node under (mx,my) in css px, or null. Discs at L1, card boxes at L2+;
   * L0 rows are subpixel — nothing to click. */
  _hitTest(mx, my) {
    const { tree, layout, tf } = this;
    const rowPx = ROW * tf.ky;
    const lod = this._lod(rowPx);
    if (lod === 0) return null;
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < tree.parent.length; i++) {
      const x = layout.x[i] * tf.kx + tf.x;
      const y = layout.y[i] * tf.ky + tf.y;
      if (lod === 1) {
        const r = Math.max(this._discRadius(i, rowPx), 7);
        const d2 = (mx - x) ** 2 + (my - y) ** 2;
        if (d2 <= r * r && d2 < bestDist) {
          best = i;
          bestDist = d2;
        }
      } else {
        const { w, h } = this._cardGeom(rowPx, lod);
        if (mx >= x - 10 && mx <= x - 10 + w && my >= y - h / 2 && my <= y + h / 2) return i;
      }
    }
    return best;
  }

  // ---- data on demand -------------------------------------------------------

  /** Ask the server for FENs (+SAN labels) of visible L3 nodes. */
  _requestFens(rows) {
    if (!this.callbacks.fetchFens || rows.length === 0) return;
    this._ensurePaths();
    const wanted = rows.filter((i) => {
      const key = this.paths[i].join(" ");
      return !this.fens.has(key) && !this.fensPending.has(key);
    });
    if (wanted.length === 0) return;
    const keys = wanted.map((i) => this.paths[i].join(" "));
    for (const key of keys) this.fensPending.add(key);
    const tree = this.tree; // guard against a base snapshot swap mid-flight
    this.callbacks.fetchFens(wanted.map((i) => this.paths[i])).then((res) => {
      if (!res || this.tree !== tree) return;
      keys.forEach((key, j) => {
        this.fensPending.delete(key);
        // cache misses too (fen: null), or every redraw re-requests them
        this.fens.set(key, { fen: res.fens[j], san: res.sans[j] });
      });
      this.draw();
    });
  }

  /** Zooming into pruned regions (§5.2): fetch the subtree of visible nodes
   * whose children were cut off by the snapshot budget. */
  _requestDetail(rows) {
    if (!this.callbacks.fetchDetail || rows.length === 0) return;
    this._ensurePaths();
    const candidates = rows
      .filter(
        (i) =>
          this.tree.visits[i] > 1 &&
          this.tree.children_total[i] > this.layout.children[i].length &&
          !this.detailDone.has(this.paths[i].join(" ")),
      )
      .sort((a, b) => this.tree.visits[b] - this.tree.visits[a])
      .slice(0, 2); // a couple per frame; the next draw picks up the rest
    for (const i of candidates) {
      const path = this.paths[i];
      this.detailDone.add(path.join(" "));
      this.callbacks.fetchDetail(path).then((msg) => this.graftDetail(path, msg));
    }
  }

  // ---- drawing --------------------------------------------------------------

  _lod(rowPx) {
    return rowPx < 3 ? 0 : rowPx < 24 ? 1 : rowPx < 120 ? 2 : 3;
  }

  _discRadius(i, rowPx) {
    return Math.min(1.2 + Math.log2(this.tree.visits[i] + 1) * 0.7, rowPx * 0.48);
  }

  _cardGeom(rowPx, lod) {
    if (lod >= 3) {
      const board = Math.min(Math.max(rowPx * 0.75, 90), 128);
      return { w: Math.max(board, 118) + 16, h: board + 46, board };
    }
    // cards stop growing at a readable size; more zoom spreads them apart
    return { w: Math.min(DX * this.tf.kx * 0.85, 128), h: Math.min(rowPx * 0.9, 66), board: 0 };
  }

  /** true if the tree's screen bounding box misses the viewport entirely */
  _offscreen(width, height) {
    const { tf, layout } = this;
    const left = tf.x;
    const right = tf.x + (layout.maxX + DX) * tf.kx;
    const top = tf.y;
    const bottom = tf.y + Math.max(layout.rows - 1, 1) * ROW * tf.ky;
    return right < 0 || left > width || bottom < 0 || top > height;
  }

  /** rows on the current PV, found by following the moves from the root */
  _pvRows() {
    const rows = new Set([0]);
    let node = 0;
    for (const uci of this.pv) {
      const next = this.layout.children[node].find((c) => this.tree.move[c] === uci);
      if (next === undefined) break;
      rows.add(next);
      node = next;
    }
    return rows;
  }

  draw() {
    const { ctx, tree, layout, tf } = this;
    const { width, height } = this.canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, width, height);
    if (!tree || tree.parent.length === 0) {
      ctx.fillStyle = "#8f8a84";
      ctx.font = "14px system-ui";
      ctx.fillText("no search tree yet — make the engine think", 24, 40);
      return;
    }

    // Never render nothing: if the whole tree is outside the viewport (the
    // user zoomed off, or panned around a tiny early tree and a snapshot
    // thousands of rows tall landed far away at the old transform), re-fit
    // and go back to auto-fit mode.
    if (width > 0 && this._offscreen(width, height)) {
      this.userMoved = false;
      this._fit();
    }

    const n = tree.parent.length;
    const rowPx = ROW * tf.ky;
    const lod = this._lod(rowPx);
    const pvRows = this._pvRows();
    const rootVisits = Math.max(tree.visits[0], 1);
    const sx = (i) => layout.x[i] * tf.kx + tf.x;
    const sy = (i) => layout.y[i] * tf.ky + tf.y;
    const margin = DX * tf.kx;
    const visible = (i) =>
      sx(i) > -margin && sx(i) < width + margin && sy(i) > -rowPx && sy(i) < height + rowPx;

    // Edges in three passes so dense regions read as density, not slabs:
    // 1. every edge as a translucent hairline (the tree's silhouette),
    // 2. well-visited edges again, width ∝ sqrt(visit share), batched into a
    //    few width buckets (one stroked path per bucket),
    // 3. the PV on top in accent color.
    const segment = (p, i) => {
      ctx.moveTo(sx(p), sy(p));
      if (lod === 0) {
        ctx.lineTo(sx(i), sy(i));
      } else {
        const mid = (sx(p) + sx(i)) / 2;
        ctx.bezierCurveTo(mid, sy(p), mid, sy(i), sx(i), sy(i));
      }
    };
    const edgeVisible = (i) => visible(i) || visible(tree.parent[i]);

    ctx.strokeStyle = "rgba(150, 145, 138, 0.08)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 1; i < n; i++) if (edgeVisible(i)) segment(tree.parent[i], i);
    ctx.stroke();

    const bucketOf = (i) =>
      Math.min(Math.round(3 * Math.sqrt(tree.visits[i] / rootVisits) * 2), 7);
    for (let bucket = 1; bucket <= 7; bucket++) {
      ctx.strokeStyle = bucket <= 2 ? "rgba(190, 185, 178, 0.35)" : "rgba(190, 185, 178, 0.6)";
      ctx.lineWidth = 0.5 + bucket / 2;
      ctx.beginPath();
      for (let i = 1; i < n; i++)
        if (bucketOf(i) === bucket && edgeVisible(i)) segment(tree.parent[i], i);
      ctx.stroke();
    }

    ctx.strokeStyle = "#6d9f4e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 1; i < n; i++)
      if (pvRows.has(i) && pvRows.has(tree.parent[i])) segment(tree.parent[i], i);
    ctx.stroke();

    if (lod === 0) return; // L0: the silhouette is the edges

    // nodes: discs (L1) or stat cards (L2/L3), colored by side to move
    const rootWhite = tree.turn === "w";
    const visibleRows = [];
    for (let i = 0; i < n; i++) {
      if (!visible(i)) continue;
      visibleRows.push(i);
      const whiteToMove = layout.depth[i] % 2 === 0 ? rootWhite : !rootWhite;
      if (lod === 1) this._disc(i, whiteToMove, pvRows.has(i), rowPx);
      else this._card(i, whiteToMove, pvRows.has(i), rowPx, lod);
    }
    if (lod >= 2) this._requestDetail(visibleRows);
    if (lod >= 3) this._requestFens(visibleRows);
  }

  _disc(i, white, onPv, rowPx) {
    const { ctx, layout, tf } = this;
    ctx.beginPath();
    ctx.arc(
      layout.x[i] * tf.kx + tf.x,
      layout.y[i] * tf.ky + tf.y,
      this._discRadius(i, rowPx),
      0,
      2 * Math.PI,
    );
    ctx.fillStyle = white ? "#f0eeeb" : "#1f1d1b";
    ctx.fill();
    ctx.lineWidth = onPv ? 1.6 : 0.8;
    ctx.strokeStyle = onPv ? "#6d9f4e" : "#8f8a84";
    ctx.stroke();
  }

  _card(i, white, onPv, rowPx, lod) {
    const { ctx, tree, tf } = this;
    const x = this.layout.x[i] * tf.kx + tf.x;
    const y = this.layout.y[i] * tf.ky + tf.y;
    const { w, h, board } = this._cardGeom(rowPx, lod);

    ctx.fillStyle = white ? "#f0eeeb" : "#262422";
    ctx.strokeStyle = onPv ? "#6d9f4e" : "#8f8a84";
    ctx.lineWidth = onPv ? 2 : 1;
    ctx.beginPath();
    ctx.roundRect(x - 10, y - h / 2, w, h, 5);
    ctx.fill();
    ctx.stroke();

    const fg = white ? "#33302c" : "#d8d5d1";
    const dim = white ? "#77716a" : "#8f8a84";
    const cached = lod >= 3 && this.paths ? this.fens.get(this.paths[i].join(" ")) : null;
    ctx.fillStyle = fg;
    ctx.font = "bold 12px system-ui";
    const label = (cached && cached.san) || tree.move[i] || "root";
    ctx.fillText(label, x - 2, y - h / 2 + 15);
    if (h >= 40) {
      ctx.fillStyle = dim;
      ctx.font = "10px system-ui";
      const winPct = (tree.q[i] * 100).toFixed(0);
      ctx.fillText(`N ${tree.visits[i]} · ${winPct}% · ${toCp(tree.q[i])}cp`, x - 2, y - h / 2 + 30);
      if (lod < 3 && h >= 54) {
        const pruned = tree.children_total[i] - this.layout.children[i].length;
        const prunedText = pruned > 0 ? `  •  +${pruned} pruned` : "";
        ctx.fillText(`P ${tree.prior[i].toFixed(3)}${prunedText}`, x - 2, y - h / 2 + 44);
      }
    }
    if (lod >= 3) this._thumbnail(cached, x - 10 + (w - board) / 2, y - h / 2 + 38, board);
  }

  /** Mini board (L3): squares always, pieces once the FEN has arrived. */
  _thumbnail(cached, x, y, size) {
    const { ctx } = this;
    const s = size / 8;
    for (let fx = 0; fx < 8; fx++)
      for (let fy = 0; fy < 8; fy++) {
        ctx.fillStyle = (fx + fy) % 2 === 0 ? "#f0d9b5" : "#b58863";
        ctx.fillRect(x + fx * s, y + fy * s, s, s);
      }
    if (!cached || !cached.fen) return;
    for (const [square, symbol] of fenPlacement(cached.fen)) {
      const file = square.charCodeAt(0) - 97;
      const rank = Number(square[1]) - 1;
      const img = pieceImage(symbol);
      if (!img.complete) {
        img.addEventListener("load", () => this.draw(), { once: true });
        continue;
      }
      ctx.drawImage(img, x + file * s, y + (7 - rank) * s, s, s);
    }
  }
}
