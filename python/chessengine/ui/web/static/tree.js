// Live search-tree view: canvas renderer with semantic zoom (DESIGN-VISU.md
// §4.1). Consumes the flat tree snapshots streamed by the server; draws
// edges only when far out (L0), side-to-move discs at mid zoom (L1) and stat
// cards up close (L2). No DOM per node — the tree has tens of thousands.

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
    }
  }
  return { x, y, depth, children, rows };
}

export class TreeView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tree = null;
    this.layout = null;
    this.pv = [];
    // world -> css px: sx = x * kx + tx. Anisotropic on purpose: the tree is
    // a few dozen plies wide but tens of thousands of rows tall, so depth and
    // rows need independent scales; zooming moves both together.
    this.tf = { x: 40, y: 40, kx: 1, ky: 1 };
    this.userMoved = false; // stop auto-fitting once the user pans/zooms

    canvas.addEventListener("wheel", (e) => this._wheel(e), { passive: false });
    canvas.addEventListener("pointerdown", (e) => this._dragStart(e));
    canvas.addEventListener("pointermove", (e) => this._dragMove(e));
    canvas.addEventListener("pointerup", () => (this.dragFrom = null));
    new ResizeObserver(() => this.refresh()).observe(canvas);
    this.refresh();
  }

  setTree(msg) {
    this.tree = msg;
    this.layout = layoutTree(msg);
    if (!this.userMoved) this._fit();
    this.draw();
  }

  setPV(pv) {
    this.pv = pv;
    if (this.tree) this.draw();
  }

  // ---- viewport -----------------------------------------------------------

  /** Re-read the canvas size and redraw. On a hidden tab the canvas is 0×0
   * and trees arriving then can't be fitted, so when the tab (re)appears the
   * view must be fitted against the now-real size. */
  refresh() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.canvas.clientWidth * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in css px below
    if (this.tree && !this.userMoved) this._fit();
    this.draw();
  }

  _fit() {
    const { width, height } = this.canvas.getBoundingClientRect();
    if (width === 0 || this.layout.rows === 0) return;
    const worldW = Math.max(...this.layout.x) + DX;
    const worldH = Math.max(this.layout.rows * ROW, ROW);
    this.tf.kx = Math.min((width - 80) / worldW, 1.3);
    this.tf.ky = Math.min((height - 80) / worldH, 1.5);
    this.tf.x = 40;
    this.tf.y = height / 2 - (worldH / 2) * this.tf.ky;
  }

  _wheel(event) {
    event.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const factor = Math.exp(-event.deltaY * 0.0015);
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

  _dragStart(event) {
    this.dragFrom = { x: event.clientX, y: event.clientY };
    this.canvas.setPointerCapture(event.pointerId);
  }

  _dragMove(event) {
    if (!this.dragFrom) return;
    this.tf.x += event.clientX - this.dragFrom.x;
    this.tf.y += event.clientY - this.dragFrom.y;
    this.dragFrom = { x: event.clientX, y: event.clientY };
    this.userMoved = true;
    this.draw();
  }

  // ---- drawing --------------------------------------------------------------

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

    const n = tree.parent.length;
    const rowPx = ROW * tf.ky;
    const lod = rowPx < 3 ? 0 : rowPx < 24 ? 1 : 2;
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

    // nodes: discs (L1) or stat cards (L2), colored by side to move
    const rootWhite = tree.turn === "w";
    for (let i = 0; i < n; i++) {
      if (!visible(i)) continue;
      const whiteToMove = layout.depth[i] % 2 === 0 ? rootWhite : !rootWhite;
      if (lod === 1) this._disc(i, whiteToMove, pvRows.has(i), rowPx);
      else this._card(i, whiteToMove, pvRows.has(i), rowPx);
    }
  }

  _disc(i, white, onPv, rowPx) {
    const { ctx, tree, layout, tf } = this;
    const r = Math.min(1.2 + Math.log2(tree.visits[i] + 1) * 0.7, rowPx * 0.48);
    ctx.beginPath();
    ctx.arc(layout.x[i] * tf.kx + tf.x, layout.y[i] * tf.ky + tf.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = white ? "#f0eeeb" : "#1f1d1b";
    ctx.fill();
    ctx.lineWidth = onPv ? 1.6 : 0.8;
    ctx.strokeStyle = onPv ? "#6d9f4e" : "#8f8a84";
    ctx.stroke();
  }

  _card(i, white, onPv, rowPx) {
    const { ctx, tree, tf } = this;
    const x = this.layout.x[i] * tf.kx + tf.x;
    const y = this.layout.y[i] * tf.ky + tf.y;
    // cards stop growing at a readable size; more zoom spreads them apart
    const w = Math.min(DX * tf.kx * 0.85, 128);
    const h = Math.min(rowPx * 0.9, 66);

    ctx.fillStyle = white ? "#f0eeeb" : "#262422";
    ctx.strokeStyle = onPv ? "#6d9f4e" : "#8f8a84";
    ctx.lineWidth = onPv ? 2 : 1;
    ctx.beginPath();
    ctx.roundRect(x - 10, y - h / 2, w, h, 5);
    ctx.fill();
    ctx.stroke();

    const fg = white ? "#33302c" : "#d8d5d1";
    const dim = white ? "#77716a" : "#8f8a84";
    ctx.fillStyle = fg;
    ctx.font = "bold 12px system-ui";
    const label = tree.move[i] || "root";
    ctx.fillText(label, x - 2, y - h / 2 + 15);
    if (h >= 40) {
      ctx.fillStyle = dim;
      ctx.font = "10px system-ui";
      const winPct = (tree.q[i] * 100).toFixed(0);
      ctx.fillText(`N ${tree.visits[i]} · ${winPct}% · ${toCp(tree.q[i])}cp`, x - 2, y - h / 2 + 30);
      if (h >= 54) {
        const pruned = tree.children_total[i] - this.layout.children[i].length;
        const prunedText = pruned > 0 ? `  •  +${pruned} pruned` : "";
        ctx.fillText(`P ${tree.prior[i].toFixed(3)}${prunedText}`, x - 2, y - h / 2 + 44);
      }
    }
  }
}
