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

const FIGURINES = {
  w: { K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙" },
  b: { K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟" },
};

/** PV edge annotation (§10.2): figurine of the moved piece + from–to. */
export function pvLabel(uci, san, whiteMoved) {
  let piece = "P";
  if (san && /^[KQRBN]/.test(san)) piece = san[0];
  else if (san && san.startsWith("O")) piece = "K"; // castling
  return `${FIGURINES[whiteMoved ? "w" : "b"][piece]} ${uci.slice(0, 2)}–${uci.slice(2, 4)}`;
}

/** Path key ("e2e4 e7e5") per row — node identity across snapshots (§5.2). */
export function pathKeys(tree) {
  const n = tree.parent.length;
  const keys = new Array(n);
  keys[0] = (tree.root_path || []).join(" ");
  for (let i = 1; i < n; i++) {
    const prefix = keys[tree.parent[i]];
    keys[i] = prefix ? `${prefix} ${tree.move[i]}` : tree.move[i];
  }
  return keys;
}

/** Shared scaffold for derived trees (collapsed/compressed): a fresh row
 * array seeded with the root, plus a pushRow() that appends a real or
 * bundle pseudo-node (bundle = {branches, sims}) and returns its row index. */
function deriveTreeBuilder(tree) {
  const out = {
    ...tree,
    parent: [-1],
    move: [""],
    visits: [tree.visits[0]],
    q: [tree.q[0]],
    prior: [tree.prior[0]],
    children_total: [tree.children_total[0]],
    bundle: [null],
  };
  const pushRow = (parent, move, visits, q, prior, childrenTotal, bundle) => {
    out.parent.push(parent);
    out.move.push(move);
    out.visits.push(visits);
    out.q.push(q);
    out.prior.push(prior);
    out.children_total.push(childrenTotal);
    out.bundle.push(bundle);
    return out.parent.length - 1;
  };
  return { out, pushRow };
}

/** Collapsed mode (§10.2): derive a tree holding only the PV chain; at each
 * PV node the branches not taken fold into one bundle pseudo-node (move "…",
 * `bundle[i]` = {branches, sims}). Simulations through the fold are computed
 * by subtraction (visits[node] − visits[pv child] − 1 own evaluation; the
 * root is never evaluated itself), so pruned-away siblings are counted too.
 */
export function collapseTree(tree, pv) {
  const n = tree.parent.length;
  const children = Array.from({ length: n }, () => []);
  for (let i = 1; i < n; i++) children[tree.parent[i]].push(i);

  const { out, pushRow } = deriveTreeBuilder(tree);
  const foldedSims = (node, keptChild) => {
    let visible = 0;
    for (const c of children[node]) if (c !== keptChild) visible += tree.visits[c];
    const kept = keptChild === null ? 0 : tree.visits[keptChild];
    const ownEval = node === 0 ? 0 : 1;
    return Math.max(tree.visits[node] - kept - ownEval, visible);
  };

  let node = 0; // full-tree row
  let outNode = 0; // its derived row
  for (const uci of pv) {
    const next = children[node].find((c) => tree.move[c] === uci);
    if (next === undefined) break;
    const branches = Math.max(tree.children_total[node] - 1, children[node].length - 1);
    if (branches > 0)
      pushRow(outNode, "…", foldedSims(node, next), 0, 0, 0, {
        branches,
        sims: foldedSims(node, next),
      });
    outNode = pushRow(
      outNode,
      tree.move[next],
      tree.visits[next],
      tree.q[next],
      tree.prior[next],
      tree.children_total[next],
      null,
    );
    node = next;
  }
  // the PV tip's own children fold too
  const tipBranches = Math.max(tree.children_total[node], children[node].length);
  if (tipBranches > 0)
    pushRow(outNode, "…", foldedSims(node, null), 0, 0, 0, {
      branches: tipBranches,
      sims: foldedSims(node, null),
    });
  return out;
}

/** Partial-compression mode: at every branching node (not just the PV),
 * keep only the top-`k` children by visit count and fold the rest into one
 * bundle pseudo-node (move "…", `bundle[i]` = {branches, sims}). `expanded`
 * is a set of path keys (§5.2) the user has clicked open — those nodes show
 * every child instead (their own children are still top-k compressed). */
export function compressTree(tree, k, expanded) {
  const n = tree.parent.length;
  const children = Array.from({ length: n }, () => []);
  for (let i = 1; i < n; i++) children[tree.parent[i]].push(i);
  const compare = SORT_CRITERIA.visits(tree);
  for (const list of children) list.sort(compare);
  const keys = pathKeys(tree);

  const { out, pushRow } = deriveTreeBuilder(tree);
  const outRow = new Array(n);
  outRow[0] = 0;
  const stack = [0];
  while (stack.length > 0) {
    const node = stack.pop();
    const kids = children[node];
    const showAll = expanded.has(keys[node]);
    const visible = showAll ? kids : kids.slice(0, k);
    for (const c of visible) {
      outRow[c] = pushRow(
        outRow[node],
        tree.move[c],
        tree.visits[c],
        tree.q[c],
        tree.prior[c],
        tree.children_total[c],
        null,
      );
      stack.push(c);
    }
    if (!showAll && kids.length > k) {
      const hidden = kids.slice(k);
      let sims = 0;
      for (const c of hidden) sims += tree.visits[c];
      const branches = Math.max(tree.children_total[node] - visible.length, hidden.length);
      pushRow(outRow[node], "…", sims, 0, 0, 0, { branches, sims });
    }
  }
  return out;
}

function fmtCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

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
    this.raw = null; // latest snapshot as sent (plus grafted detail)
    this.tree = null; // what is drawn: raw, or its collapsed derivation
    this.layout = null;
    this.pv = [];
    this.pvSan = [];
    this.collapsed = false; // §10.2 collapsed mode
    this.compressed = false; // partial-compression mode: top-k children per node
    this.compressK = 4;
    this.expandedBundles = new Set(); // path keys the user expanded (compressed mode)
    this.cPuct = 1.5; // for the hover PUCT breakdown; app updates from config
    this.paths = null; // paths[i]: UCI move path from the game position
    this.pathIndex = null; // path key ("e2e4 e7e5") -> row
    this.fens = new Map(); // path key -> {fen, san} for L3 thumbnails
    this.fensPending = new Set();
    this.detailDone = new Set(); // subtrees already requested this snapshot
    this.stepBaseline = null; // path key -> visits before the last step (§10.4)
    this.hoverEnabled = false;
    this.hoverPos = null;
    this.pendingClick = null; // armed explore-click, cancelled by a double click
    this.lastClick = null;
    // world -> css px: sx = x * kx + tx. Anisotropic on purpose: the tree is
    // a few dozen plies wide but tens of thousands of rows tall, so depth and
    // rows need independent scales; zooming moves both together.
    this.tf = { x: 40, y: 40, kx: 1, ky: 1 };
    this.userMoved = false; // stop auto-fitting once the user pans/zooms

    this.tip = this._overlay("tree-tip"); // hover tooltip (§10.2)
    this.nav = this._overlay("tree-nav"); // navigation chips (§10.2)
    this._buildNav();

    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("pointerdown", (e) => this._dragStart(e));
    canvas.addEventListener("pointermove", (e) => this._dragMove(e));
    canvas.addEventListener("pointerup", (e) => this._dragEnd(e));
    canvas.addEventListener("pointerleave", () => this._hideTip());
    new ResizeObserver(() => this.refresh()).observe(canvas);
    this.refresh();
  }

  _overlay(id) {
    const el = document.createElement("div");
    el.id = id;
    el.hidden = true;
    this.canvas.parentElement.appendChild(el);
    return el;
  }

  setTree(msg) {
    // a new base position invalidates expand state from the old tree's paths
    if (this.raw && this.raw.fen !== msg.fen) this.expandedBundles.clear();
    this.raw = msg;
    this.fens.clear();
    this.fensPending.clear();
    this._rebuild();
  }

  setPV(pv, pvSan = []) {
    this.pv = pv;
    this.pvSan = pvSan;
    if (!this.raw) return;
    if (this.collapsed) this._rebuild();
    else this.draw();
  }

  /** Toggle collapsed mode (§10.2) and refit — the shape changes entirely. */
  setCollapsed(on) {
    if (this.collapsed === on) return;
    this.collapsed = on;
    if (on && this.compressed) {
      this.compressed = false;
      this.callbacks.onCompressChange?.(false);
    }
    this.callbacks.onCollapseChange?.(on);
    this.userMoved = false;
    if (this.raw) this._rebuild();
  }

  /** Toggle partial-compression mode: top-`compressK` children per node,
   * the rest folded into a collector bundle (click to expand it in place). */
  setCompressed(on) {
    if (this.compressed === on) return;
    this.compressed = on;
    if (on && this.collapsed) {
      this.collapsed = false;
      this.callbacks.onCollapseChange?.(false);
    }
    this.callbacks.onCompressChange?.(on);
    this.userMoved = false;
    if (this.raw) this._rebuild();
  }

  /** Change K (children kept per node) and re-derive if compression is active. */
  setCompressK(k) {
    const v = Math.max(1, Math.round(k) || 4);
    if (this.compressK === v) return;
    this.compressK = v;
    if (this.compressed && this.raw) this._rebuild();
  }

  /** Un-fold one node's collector bundle in place (§ compressed mode). */
  _expandBundle(hit) {
    this._ensurePaths();
    const key = this.paths[this.tree.parent[hit]].join(" ");
    this.expandedBundles.add(key);
    this._rebuild();
  }

  /** Capture pre-step visit counts; draw() highlights what a step changed. */
  markStep() {
    this.stepBaseline = new Map();
    // path keys only compare within one base position — if the tree gets
    // re-rooted (engine move) the diff is meaningless and must stay dark
    this.stepFen = this.raw ? this.raw.fen : null;
    if (!this.raw) return;
    const keys = pathKeys(this.raw);
    for (let i = 0; i < keys.length; i++) this.stepBaseline.set(keys[i], this.raw.visits[i]);
  }

  clearStep() {
    if (!this.stepBaseline) return;
    this.stepBaseline = null;
    if (this.tree) this.draw();
  }

  /** Recompute the drawn tree (raw or collapsed) and its layout. */
  _rebuild() {
    this.tree = this.collapsed
      ? collapseTree(this.raw, this.pv)
      : this.compressed
        ? compressTree(this.raw, this.compressK, this.expandedBundles)
        : this.raw;
    this.layout = layoutTree(this.tree);
    this.paths = null;
    this.pathIndex = null;
    this.detailDone.clear();
    if (!this.userMoved) this._fit();
    this.draw();
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
    // while collapsed/compressed the drawn tree is derived — don't graft into
    // it (any in-flight response is applied via the next request instead)
    if (this.collapsed || this.compressed || !this.tree || !msg || msg.parent.length <= 1) return;
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
    this._hideTip();
    this.canvas.setPointerCapture(event.pointerId);
  }

  _dragMove(event) {
    if (!this.dragFrom) {
      if (this.hoverEnabled) {
        const rect = this.canvas.getBoundingClientRect();
        this.hoverPos = { mx: event.clientX - rect.left, my: event.clientY - rect.top };
        if (!this.hoverRaf)
          this.hoverRaf = requestAnimationFrame(() => {
            this.hoverRaf = null;
            this._updateHover();
          });
      }
      return;
    }
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

  // ---- click-to-explore (§4.2) + double-click zoom (§10.2) ------------------

  _click(event) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const prev = this.lastClick;
    this.lastClick = { t: performance.now(), x: mx, y: my };
    if (prev && this.lastClick.t - prev.t < 300 && Math.hypot(mx - prev.x, my - prev.y) < 12) {
      // double click: zoom toward the cursor, cancel any armed explore
      clearTimeout(this.pendingClick);
      this.pendingClick = null;
      this.lastClick = null;
      this._zoomAt(mx, my, 2);
      return;
    }
    if (!this.tree || !this.callbacks.onNodeClick) return;
    const hit = this._hitTest(mx, my);
    if (hit === null) return;
    if (this.tree.bundle && this.tree.bundle[hit]) {
      if (this.collapsed) return this.setCollapsed(false);
      if (this.compressed) return this._expandBundle(hit);
      return;
    }
    if (hit === 0) return; // root = current position, no-op
    this._ensurePaths();
    const path = this.paths[hit];
    // armed with a delay so a double click can cancel it (it would otherwise
    // open the explore confirm dialog)
    this.pendingClick = setTimeout(() => {
      this.pendingClick = null;
      this.callbacks.onNodeClick(path);
    }, 260);
  }

  /** Node under (mx,my) in css px, or null. Discs at L0/L1 (even sub-pixel
   * rows keep a fixed hoverable radius), card boxes at L2+. */
  _hitTest(mx, my) {
    const { tree, layout, tf } = this;
    const rowPx = ROW * tf.ky;
    const lod = this._lod(rowPx);
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < tree.parent.length; i++) {
      const x = layout.x[i] * tf.kx + tf.x;
      const y = layout.y[i] * tf.ky + tf.y;
      if (lod <= 1) {
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

  /** Ask the server for FENs (+SAN labels) of visible L3 nodes, replayed
   * from this snapshot's own base position (§10.1). */
  _requestFens(rows) {
    if (!this.callbacks.fetchFens || rows.length === 0) return;
    this._ensurePaths();
    const wanted = rows.filter((i) => {
      if (this.tree.bundle && this.tree.bundle[i]) return false; // no position
      const key = this.paths[i].join(" ");
      return !this.fens.has(key) && !this.fensPending.has(key);
    });
    if (wanted.length === 0) return;
    const keys = wanted.map((i) => this.paths[i].join(" "));
    for (const key of keys) this.fensPending.add(key);
    const raw = this.raw; // guard against a base snapshot swap mid-flight
    this.callbacks.fetchFens(
      wanted.map((i) => this.paths[i]),
      raw.fen,
    ).then((res) => {
      if (!res || this.raw !== raw) return;
      keys.forEach((key, j) => {
        this.fensPending.delete(key);
        // cache misses too (fen: null), or every redraw re-requests them
        this.fens.set(key, { fen: res.fens[j], san: res.sans[j] });
      });
      this.draw();
      if (this.hoverEnabled && this.hoverPos) this._updateHover();
    });
  }

  /** Zooming into pruned regions (§5.2): fetch the subtree of visible nodes
   * whose children were cut off by the snapshot budget. */
  _requestDetail(rows) {
    if (this.collapsed || this.compressed) return; // the fold is intentional, don't unfold it
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

  /** rows on the current PV in order, found by following moves from the root */
  _pvChain() {
    const chain = [0];
    let node = 0;
    for (const uci of this.pv) {
      const next = this.layout.children[node].find((c) => this.tree.move[c] === uci);
      if (next === undefined) break;
      chain.push(next);
      node = next;
    }
    return chain;
  }

  /** did the last step (§10.4) touch this node? — backprop increments the
   * visits of exactly the traversed path, so a visit diff IS the path */
  _stepChanged(i) {
    if (!this.stepBaseline || this.tree.fen !== this.stepFen) return false;
    if (this.tree.bundle && this.tree.bundle[i]) return false;
    this._ensurePaths();
    return this.stepBaseline.get(this.paths[i].join(" ")) !== this.tree.visits[i];
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
    const pvChain = this._pvChain();
    const pvRows = new Set(pvChain);
    const rootWhite = tree.turn === "w";
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

    ctx.strokeStyle = "rgba(200, 196, 190, 0.18)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 1; i < n; i++) if (edgeVisible(i)) segment(tree.parent[i], i);
    ctx.stroke();

    const bucketOf = (i) =>
      Math.min(Math.round(3 * Math.sqrt(tree.visits[i] / rootVisits) * 2), 7);
    for (let bucket = 1; bucket <= 7; bucket++) {
      ctx.strokeStyle = bucket <= 2 ? "rgba(212, 208, 202, 0.45)" : "rgba(212, 208, 202, 0.75)";
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

    if (lod === 0) {
      // §10.2: never edges-only — every node at least a tiny b/w dot
      const r = Math.max(rowPx * 0.42, 0.8);
      for (const white of [true, false]) {
        ctx.fillStyle = white ? "#f0eeeb" : "#1f1d1b";
        ctx.strokeStyle = "#8f8a84";
        ctx.lineWidth = 0.3;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          if (!visible(i)) continue;
          const whiteToMove = layout.depth[i] % 2 === 0 ? rootWhite : !rootWhite;
          if (whiteToMove !== white) continue;
          ctx.moveTo(sx(i) + r, sy(i));
          ctx.arc(sx(i), sy(i), r, 0, 2 * Math.PI);
        }
        ctx.fill();
        ctx.stroke();
      }
      this._updateNav(null);
      return;
    }

    // nodes: discs (L1), stat cards (L2/L3) colored by side to move, or
    // bundle pseudo-nodes in collapsed mode
    const visibleRows = [];
    for (let i = 0; i < n; i++) {
      if (!visible(i)) continue;
      visibleRows.push(i);
      const whiteToMove = layout.depth[i] % 2 === 0 ? rootWhite : !rootWhite;
      if (tree.bundle && tree.bundle[i]) this._bundle(i, rowPx, lod);
      else if (lod === 1) this._disc(i, whiteToMove, pvRows.has(i), rowPx);
      else this._card(i, whiteToMove, pvRows.has(i), rowPx, lod);
    }
    // PV move annotations (§10.2): figurine + from–to along each PV edge,
    // once there is horizontal room. Drawn on top: at dense zoom the label
    // spots inevitably overlap the cards, which must not cover them.
    if (DX * tf.kx > 55) {
      ctx.font = "11px system-ui";
      for (let k = 1; k < pvChain.length && k - 1 < this.pv.length; k++) {
        const whiteMoved = k % 2 === 1 ? rootWhite : !rootWhite;
        const label = pvLabel(this.pv[k - 1], this.pvSan[k - 1], whiteMoved);
        const mx = (sx(pvChain[k - 1]) + sx(pvChain[k])) / 2;
        const my = (sy(pvChain[k - 1]) + sy(pvChain[k])) / 2;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(38, 36, 33, 0.88)"; // backing chip for contrast
        ctx.fillRect(mx - tw / 2 - 3, my - 15, tw + 6, 14);
        ctx.fillStyle = "#9fc97e";
        ctx.fillText(label, mx - tw / 2, my - 4);
      }
    }

    if (lod >= 2) this._requestDetail(visibleRows);
    if (lod >= 3) this._requestFens(visibleRows);
    this._updateNav(lod >= 2 ? visibleRows : null, width, height);
  }

  _disc(i, white, onPv, rowPx) {
    const { ctx, layout, tf } = this;
    const stepped = this._stepChanged(i);
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
    ctx.lineWidth = stepped ? 2 : onPv ? 1.6 : 0.8;
    ctx.strokeStyle = stepped ? "#e8a33d" : onPv ? "#6d9f4e" : "#8f8a84";
    ctx.stroke();
  }

  /** Folded siblings, stacked pseudo-node: the branches not on the PV
   * (collapsed mode, click un-collapses) or beyond top-K (compressed mode,
   * click expands that node in place). */
  _bundle(i, rowPx, lod) {
    const { ctx, tree, tf } = this;
    const x = this.layout.x[i] * tf.kx + tf.x;
    const y = this.layout.y[i] * tf.ky + tf.y;
    const { branches, sims } = tree.bundle[i];
    const w = lod === 1 ? 26 : Math.min(DX * tf.kx * 0.85, 128);
    const h = lod === 1 ? 18 : Math.min(Math.max(rowPx * 0.5, 34), 46);
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#8f8a84";
    ctx.fillStyle = "#2c2a27";
    ctx.lineWidth = 1;
    // stacked look: a second outline peeking out behind the card
    ctx.beginPath();
    ctx.roundRect(x - 10 + 4, y - h / 2 + 4, w, h, 5);
    ctx.stroke();
    ctx.beginPath();
    ctx.roundRect(x - 10, y - h / 2, w, h, 5);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
    if (lod >= 2) {
      ctx.fillStyle = "#d8d5d1";
      ctx.font = "bold 11px system-ui";
      ctx.fillText(`⑂ ${branches} branch${branches === 1 ? "" : "es"}`, x - 2, y - h / 2 + 15);
      ctx.fillStyle = "#8f8a84";
      ctx.font = "10px system-ui";
      ctx.fillText(`${fmtCount(sims)} sims`, x - 2, y - h / 2 + 29);
    }
  }

  _card(i, white, onPv, rowPx, lod) {
    const { ctx, tree, tf } = this;
    const x = this.layout.x[i] * tf.kx + tf.x;
    const y = this.layout.y[i] * tf.ky + tf.y;
    const { w, h, board } = this._cardGeom(rowPx, lod);
    const stepped = this._stepChanged(i);

    ctx.fillStyle = white ? "#f0eeeb" : "#262422";
    ctx.strokeStyle = stepped ? "#e8a33d" : onPv ? "#6d9f4e" : "#8f8a84";
    ctx.lineWidth = stepped || onPv ? 2 : 1;
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

  /** Mini board (L3 cards and hover tips): squares always, pieces once the
   * FEN has arrived. */
  _thumbnail(cached, x, y, size, ctx = this.ctx) {
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

  // ---- hover info (§10.2) ----------------------------------------------------

  setHoverEnabled(on) {
    this.hoverEnabled = on;
    if (!on) this._hideTip();
  }

  _updateHover() {
    if (!this.hoverEnabled || !this.hoverPos || !this.tree) return this._hideTip();
    const { mx, my } = this.hoverPos;
    const node = this._hitTest(mx, my);
    if (node !== null) return this._nodeTip(node, mx, my);
    const edge = this._edgeHit(mx, my);
    if (edge !== null) return this._edgeTip(edge, mx, my);
    this._hideTip();
  }

  /** Nearest edge within ~6 css px of (mx,my); the bezier bow is inside the
   * tolerance, so the parent→child straight segment is close enough. */
  _edgeHit(mx, my) {
    const { tree, layout, tf } = this;
    const sx = (i) => layout.x[i] * tf.kx + tf.x;
    const sy = (i) => layout.y[i] * tf.ky + tf.y;
    let best = null;
    let bestD = 6;
    for (let i = 1; i < tree.parent.length; i++) {
      const p = tree.parent[i];
      const x1 = sx(p), y1 = sy(p), x2 = sx(i), y2 = sy(i);
      if (mx < Math.min(x1, x2) - 6 || mx > Math.max(x1, x2) + 6) continue;
      if (my < Math.min(y1, y2) - 6 || my > Math.max(y1, y2) + 6) continue;
      const lenSq = (x2 - x1) ** 2 + (y2 - y1) ** 2;
      const t = lenSq === 0 ? 0 : Math.min(Math.max(((mx - x1) * (x2 - x1) + (my - y1) * (y2 - y1)) / lenSq, 0), 1);
      const d = Math.hypot(mx - (x1 + t * (x2 - x1)), my - (y1 + t * (y2 - y1)));
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  _nodeTip(i, mx, my) {
    const { tree } = this;
    if (tree.bundle && tree.bundle[i]) {
      const { branches, sims } = tree.bundle[i];
      this._setTip(
        `<div class="tip-title">⑂ ${branches} folded branch${branches === 1 ? "" : "es"}</div>` +
          `<div class="tip-line">${fmtCount(sims)} simulations ran through them</div>`,
        mx,
        my,
      );
      return;
    }
    this._ensurePaths();
    const key = this.paths[i].join(" ");
    const cached = this.fens.get(key);
    if (!cached) this._requestFens([i]); // tooltip re-renders when it arrives
    const parent = tree.parent[i];
    // the PUCT terms selection sees: Q (win frequency) + U (exploration)
    const u =
      parent >= 0
        ? (this.cPuct * tree.prior[i] * Math.sqrt(Math.max(tree.visits[parent], 1))) /
          (1 + tree.visits[i])
        : 0;
    const title = (cached && cached.san) || tree.move[i] || "root";
    const winPct = (tree.q[i] * 100).toFixed(1);
    const lines = [
      `<div class="tip-title">${title}</div>`,
      `<div class="tip-line">N ${fmtCount(tree.visits[i])} · ${winPct}% · ${toCp(tree.q[i])}cp</div>`,
      parent >= 0
        ? `<div class="tip-line">P ${tree.prior[i].toFixed(3)} · Q ${tree.q[i].toFixed(3)} + U ${u.toFixed(3)}</div>`
        : "",
      cached && cached.fen ? `<canvas width="96" height="96"></canvas>` : "",
    ];
    this._setTip(lines.join(""), mx, my);
    const board = this.tip.querySelector("canvas");
    if (board) this._thumbnail(cached, 0, 0, 96, board.getContext("2d"));
  }

  _edgeTip(i, mx, my) {
    if (this.tree.bundle && this.tree.bundle[i]) return this._nodeTip(i, mx, my);
    this._ensurePaths();
    const moves = this.paths[i];
    // SAN where the fens cache already knows it, UCI otherwise
    const parts = moves.map((uci, j) => {
      const cached = this.fens.get(moves.slice(0, j + 1).join(" "));
      return (cached && cached.san) || uci;
    });
    this._setTip(
      `<div class="tip-title">line (${moves.length} pl${moves.length === 1 ? "y" : "ies"})</div>` +
        `<div class="tip-line">${parts.join(" ")}</div>`,
      mx,
      my,
    );
  }

  _setTip(html, mx, my) {
    const { width } = this.canvas.getBoundingClientRect();
    this.tip.innerHTML = html;
    this.tip.hidden = false;
    // place beside the cursor, flipping to the left near the right edge
    if (mx > width * 0.65) {
      this.tip.style.left = "auto";
      this.tip.style.right = `${width - mx + 14}px`;
    } else {
      this.tip.style.right = "auto";
      this.tip.style.left = `${mx + 14}px`;
    }
    this.tip.style.top = `${my + 12}px`;
  }

  _hideTip() {
    this.tip.hidden = true;
  }

  // ---- navigation chips (§10.2) -----------------------------------------------

  _buildNav() {
    this.nav.innerHTML =
      `<span class="nav-cur"></span>` +
      `<button data-nav="root" title="jump to the root">⌂</button>` +
      `<button data-nav="parent" title="previous move (parent)">←</button>` +
      `<button data-nav="best" title="best move at the previous node">★</button>` +
      `<button data-nav="better" title="next better sibling">↑</button>` +
      `<button data-nav="worse" title="next worse sibling">↓</button>`;
    this.navTargets = {};
    for (const btn of this.nav.querySelectorAll("button"))
      btn.addEventListener("click", () => {
        const row = this.navTargets[btn.dataset.nav];
        if (row !== undefined) this._panToRow(row);
      });
  }

  /** Pan (zoom preserved) so `row` sits at the viewport center. */
  _panToRow(row) {
    const { width, height } = this.canvas.getBoundingClientRect();
    this.tf.x = width / 2 - this.layout.x[row] * this.tf.kx;
    this.tf.y = height / 2 - this.layout.y[row] * this.tf.ky;
    this.userMoved = true;
    this.draw();
  }

  /** Refresh the chips for the visible node nearest the viewport center;
   * hidden below card zoom (visibleRows = null). */
  _updateNav(visibleRows, width, height) {
    if (!visibleRows || visibleRows.length === 0) {
      this.nav.hidden = true;
      return;
    }
    const { tree, layout, tf } = this;
    let cur = visibleRows[0];
    let bestD = Infinity;
    for (const i of visibleRows) {
      const d =
        (layout.x[i] * tf.kx + tf.x - width / 2) ** 2 +
        (layout.y[i] * tf.ky + tf.y - height / 2) ** 2;
      if (d < bestD) {
        bestD = d;
        cur = i;
      }
    }
    const parent = tree.parent[cur];
    const siblings = parent >= 0 ? layout.children[parent] : []; // sorted by visits
    const rank = siblings.indexOf(cur);
    this.navTargets = {
      root: cur !== 0 ? 0 : undefined,
      parent: parent >= 0 ? parent : undefined,
      best: rank > 0 ? siblings[0] : undefined, // already on it when rank 0
      better: rank > 0 ? siblings[rank - 1] : undefined,
      worse: rank >= 0 && rank < siblings.length - 1 ? siblings[rank + 1] : undefined,
    };
    if (Object.values(this.navTargets).every((t) => t === undefined)) {
      this.nav.hidden = true; // bare root: nowhere to jump
      return;
    }
    const isBundle = tree.bundle && tree.bundle[cur];
    this.nav.querySelector(".nav-cur").textContent = isBundle
      ? "⑂"
      : tree.move[cur] || "root";
    for (const btn of this.nav.querySelectorAll("button"))
      btn.disabled = this.navTargets[btn.dataset.nav] === undefined;
    this.nav.hidden = false;
  }
}
