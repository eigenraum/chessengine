// Pure-function tests for the frontend (DESIGN-VISU.md §8): layout, FEN
// helpers. Run with: node --test tests/js
import { test } from "node:test";
import assert from "node:assert/strict";

import { layoutTree } from "../../python/chessengine/ui/web/static/tree.js";
import { fenPlacement } from "../../python/chessengine/ui/web/static/board.js";
import { composeFen, castlingAvailable } from "../../python/chessengine/ui/web/static/edit.js";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

test("layoutTree: depths, rows and parent centering", () => {
  //      0 ── 1 ── 2
  //        └─ 3
  const tree = {
    parent: [-1, 0, 1, 0],
    move: ["", "e2e4", "e7e5", "d2d4"],
    visits: [100, 60, 40, 30],
  };
  const layout = layoutTree(tree);
  assert.deepEqual([...layout.depth], [0, 1, 2, 1]);
  assert.equal(layout.rows, 2); // two leaves: node 2 and node 3
  assert.equal(layout.x[2], 2 * layout.x[1]); // x is linear in depth
  // parent sits centered on its children
  assert.equal(layout.y[0], (layout.y[1] + layout.y[3]) / 2);
});

test("layoutTree: siblings sorted by visit count, most visited on top", () => {
  const tree = {
    parent: [-1, 0, 0, 0],
    move: ["", "a", "b", "c"],
    visits: [10, 2, 7, 1],
  };
  const layout = layoutTree(tree);
  // node 2 (7 visits) above node 1 (2) above node 3 (1)
  assert.ok(layout.y[2] < layout.y[1] && layout.y[1] < layout.y[3]);
});

test("fenPlacement parses the start position", () => {
  const placement = fenPlacement(START_FEN);
  assert.equal(placement.size, 32);
  assert.equal(placement.get("e1"), "K");
  assert.equal(placement.get("e8"), "k");
  assert.equal(placement.get("a2"), "P");
  assert.equal(placement.get("h7"), "p");
});

test("composeFen round-trips through fenPlacement", () => {
  const placement = fenPlacement(START_FEN);
  const fen = composeFen(placement, "w", "KQkq");
  assert.equal(fen, START_FEN);
  assert.deepEqual(fenPlacement(fen), placement);
});

test("composeFen: empty board and empty castling", () => {
  assert.equal(composeFen(new Map(), "b", ""), "8/8/8/8/8/8/8/8 b - - 0 1");
});

test("castlingAvailable derives rights from king/rook home squares", () => {
  assert.deepEqual(castlingAvailable(fenPlacement(START_FEN)), {
    K: true,
    Q: true,
    k: true,
    q: true,
  });
  const noWhiteKingside = fenPlacement("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1");
  assert.deepEqual(castlingAvailable(noWhiteKingside), { K: false, Q: true, k: true, q: true });
});
