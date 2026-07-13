// Eval-history sparkline (§11.2): pure layout math, node-testable; the
// canvas rendering and the click-to-takeback live in app.js.

/**
 * Map eval entries onto canvas coordinates: x = ply, y = white win
 * probability (1.0 at the top).
 * @param {{ply: number, white_win_prob: number}[]} entries sorted by ply
 * @param {number} plySpan x-axis extent in plies (>= max entry ply)
 * @param {number} w @param {number} h canvas size in CSS px
 * @returns {{x: number, y: number, ply: number}[]}
 */
export function sparklinePoints(entries, plySpan, w, h, pad = 4) {
  const span = Math.max(plySpan, 1);
  return entries.map((e) => ({
    ply: e.ply,
    x: pad + (e.ply / span) * (w - 2 * pad),
    y: pad + (1 - e.white_win_prob) * (h - 2 * pad),
  }));
}

/** Ply of the point nearest to click-x, or null when none is within reach. */
export function nearestPly(points, x, radius = 10) {
  let best = null;
  for (const p of points) {
    const d = Math.abs(p.x - x);
    if (d <= radius && (best === null || d < best.d)) best = { d, ply: p.ply };
  }
  return best === null ? null : best.ply;
}
