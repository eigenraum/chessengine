// V4 pure-function tests (DESIGN-VISU.md §10.2): collapsed-mode derivation,
// PV move labels, path keys. Run with: node --test 'tests/js/*.test.mjs'
import { test } from "node:test";
import assert from "node:assert/strict";

import { collapseTree, pvLabel, pathKeys } from "../../python/chessengine/ui/web/static/tree.js";

// root ── e2e4 ── e7e5 ── g1f3
//      │       └─ c7c5
//      └─ d2d4
// visits: each node = 1 (own eval, root exempt) + subtree sims through it
const TREE = {
  turn: "w",
  fen: "startpos-fen",
  root_path: [],
  parent: [-1, 0, 1, 2, 1, 0],
  move: ["", "e2e4", "e7e5", "g1f3", "c7c5", "d2d4"],
  visits: [100, 70, 40, 12, 25, 30],
  q: [0.5, 0.55, 0.45, 0.52, 0.4, 0.35],
  prior: [1, 0.4, 0.5, 0.3, 0.3, 0.3],
  children_total: [5, 8, 6, 0, 0, 0],
};

test("collapseTree keeps the PV chain and folds siblings into bundles", () => {
  const c = collapseTree(TREE, ["e2e4", "e7e5"]);
  // root, bundle@root, e2e4, bundle@e2e4, e7e5, tip bundle
  assert.deepEqual(c.move, ["", "…", "e2e4", "…", "e7e5", "…"]);
  assert.deepEqual(c.parent, [-1, 0, 0, 2, 2, 4]);
  // PV nodes keep their stats; bundles carry {branches, sims}
  assert.equal(c.visits[2], 70);
  assert.equal(c.bundle[2], null);
  // at the root: children_total 5, one taken -> 4 folded; sims = 100 - 70
  // (the root itself is never evaluated)
  assert.deepEqual(c.bundle[1], { branches: 4, sims: 30 });
  // at e2e4: 8 children total, one taken -> 7; sims = 70 - 40 - 1 own eval
  assert.deepEqual(c.bundle[3], { branches: 7, sims: 29 });
  // the PV tip's children all fold: 6 branches, 40 - 1 own eval
  assert.deepEqual(c.bundle[5], { branches: 6, sims: 39 });
});

test("collapseTree: folded sims never undercount the visible siblings", () => {
  // stale/racy snapshot: parent visits lag behind the children's sum
  const racy = { ...TREE, visits: [100, 50, 40, 12, 25, 30] };
  const c = collapseTree(racy, ["e2e4", "e7e5"]);
  // subtraction would give 50 - 40 - 1 = 9, but c7c5 alone shows 25
  assert.equal(c.bundle[3].sims, 25);
});

test("collapseTree stops at a PV move missing from the snapshot", () => {
  const c = collapseTree(TREE, ["e2e4", "b8c6"]);
  // chain ends at e2e4; ALL its children fold into the tip bundle
  // (8 branches; 70 visits - 1 own eval = 69 sims ran below it)
  assert.deepEqual(c.move, ["", "…", "e2e4", "…"]);
  assert.deepEqual(c.bundle[3], { branches: 8, sims: 69 });
});

test("pvLabel: figurine by mover and piece, from–to from the UCI", () => {
  assert.equal(pvLabel("g1f3", "Nf3", true), "♘ g1–f3");
  assert.equal(pvLabel("g8f6", "Nf6", false), "♞ g8–f6");
  assert.equal(pvLabel("e2e4", "e4", true), "♙ e2–e4"); // pawn: no SAN letter
  assert.equal(pvLabel("e1g1", "O-O", true), "♔ e1–g1"); // castling
  assert.equal(pvLabel("d7d8q", "d8=Q+", false), "♟ d7–d8"); // promotion moves the pawn
});

test("pathKeys builds identity keys, honoring the snapshot's root_path", () => {
  assert.deepEqual(pathKeys(TREE).slice(0, 4), ["", "e2e4", "e2e4 e7e5", "e2e4 e7e5 g1f3"]);
  const sub = { ...TREE, root_path: ["d2d4", "d7d5"] };
  assert.equal(pathKeys(sub)[2], "d2d4 d7d5 e2e4 e7e5");
});
