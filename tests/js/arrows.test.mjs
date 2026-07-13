// PV arrow geometry (§11.1): pure polygon math from board.js.
import test from "node:test";
import assert from "node:assert/strict";

import { arrowPolygon } from "../../python/chessengine/ui/web/static/board.js";

const SQ = 45;

test("arrow tip is the target square center", () => {
  // e2 -> e4 with white orientation: straight up, two squares
  const pts = arrowPolygon({ x: 4 * SQ, y: 6 * SQ }, { x: 4 * SQ, y: 4 * SQ });
  assert.equal(pts.length, 7);
  const [tipX, tipY] = pts[3];
  assert.ok(Math.abs(tipX - (4 * SQ + SQ / 2)) < 1e-9);
  assert.ok(Math.abs(tipY - (4 * SQ + SQ / 2)) < 1e-9);
});

test("vertical arrow is symmetric around its axis", () => {
  const pts = arrowPolygon({ x: 0, y: 4 * SQ }, { x: 0, y: 0 });
  const axis = SQ / 2;
  // points k and (6 - k) mirror each other; the tip (index 3) is on the axis
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(pts[k][0] - axis + (pts[6 - k][0] - axis)) < 1e-9);
    assert.ok(Math.abs(pts[k][1] - pts[6 - k][1]) < 1e-9);
  }
});

test("head is wider than the shaft, shaft starts off the source center", () => {
  const pts = arrowPolygon({ x: 0, y: 0 }, { x: 3 * SQ, y: 0 });
  const axis = SQ / 2; // horizontal arrow: axis is y = 22.5
  const shaftHalf = Math.abs(pts[0][1] - axis);
  const headHalf = Math.abs(pts[2][1] - axis);
  assert.ok(headHalf > shaftHalf * 2);
  assert.ok(pts[0][0] > SQ / 2); // tail leaves the source piece visible
});

test("diagonal (knight-like) arrows keep the tip exact", () => {
  const pts = arrowPolygon({ x: 6 * SQ, y: 7 * SQ }, { x: 5 * SQ, y: 5 * SQ });
  assert.deepEqual(pts[3], [5 * SQ + SQ / 2, 5 * SQ + SQ / 2]);
});
