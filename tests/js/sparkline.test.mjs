// Eval sparkline layout (§11.2): ply -> x, win prob -> y, click hit test.
import test from "node:test";
import assert from "node:assert/strict";

import { sparklinePoints, nearestPly } from "../../python/chessengine/ui/web/static/sparkline.js";

test("points map ply to x and win prob to y (1.0 on top)", () => {
  const entries = [
    { ply: 0, white_win_prob: 0.5 },
    { ply: 6, white_win_prob: 1.0 },
    { ply: 12, white_win_prob: 0.0 },
  ];
  const pts = sparklinePoints(entries, 12, 104, 58, 4); // inner box 96 x 50
  assert.deepEqual(
    pts,
    [
      { ply: 0, x: 4, y: 29 }, // 0.5 -> vertical center
      { ply: 6, x: 52, y: 4 }, // certain white win -> top padding
      { ply: 12, x: 100, y: 54 }, // certain loss -> bottom
    ],
  );
});

test("span smaller than the max ply never divides by zero", () => {
  const pts = sparklinePoints([{ ply: 0, white_win_prob: 0.5 }], 0, 100, 50);
  assert.equal(pts[0].x, 4); // span clamps to 1
});

test("nearestPly picks the closest point within the radius", () => {
  const pts = sparklinePoints(
    [
      { ply: 0, white_win_prob: 0.5 },
      { ply: 4, white_win_prob: 0.6 },
    ],
    8,
    104,
    58,
  ); // x = 4 and 52
  assert.equal(nearestPly(pts, 50), 4);
  assert.equal(nearestPly(pts, 6), 0);
  assert.equal(nearestPly(pts, 80), null); // out of reach
  assert.equal(nearestPly([], 10), null);
});
