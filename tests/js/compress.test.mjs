// Partial-compression mode tests: top-K children per node, folded into a
// collector bundle; expanding a bundle reveals that node's real children.
// Run with: node --test 'tests/js/*.test.mjs'
import { test } from "node:test";
import assert from "node:assert/strict";

import { compressTree } from "../../python/chessengine/ui/web/static/tree.js";

// root has 6 children (visits 40/30/15/8/4/2); "a" itself has 3 children.
const TREE = {
  turn: "w",
  fen: "startpos-fen",
  root_path: [],
  parent: [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
  move: ["", "a", "b", "c", "d", "e", "f", "a1", "a2", "a3"],
  visits: [100, 40, 30, 15, 8, 4, 2, 20, 15, 4],
  q: [0.5, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.7, 0.6, 0.4],
  prior: [0, 0.3, 0.25, 0.2, 0.1, 0.1, 0.05, 0.4, 0.3, 0.2],
  children_total: [6, 3, 0, 0, 0, 0, 0, 0, 0, 0],
};

test("compressTree keeps the top-k children by visits and folds the rest", () => {
  const c = compressTree(TREE, 4, new Set());
  // root: a(40) b(30) c(15) d(8) kept, e(4) f(2) folded
  assert.deepEqual(c.move, ["", "a", "b", "c", "d", "…", "a1", "a2", "a3"]);
  assert.deepEqual(c.bundle[5], { branches: 2, sims: 6 });
  // "a" has only 3 children, all fit under k=4 — no fold there
  assert.deepEqual(c.move.slice(6), ["a1", "a2", "a3"]);
});

test("compressTree: an empty expanded set folds every over-k node", () => {
  const c = compressTree(TREE, 2, new Set());
  // root keeps only a(40), b(30); c,d,e,f (4 branches, 15+8+4+2=29 sims) fold
  assert.deepEqual(c.move.slice(0, 4), ["", "a", "b", "…"]);
  assert.deepEqual(c.bundle[3], { branches: 4, sims: 29 });
  // "a" also over k=2: a1(20), a2(15) kept, a3(4) folded
  const aRow = c.move.indexOf("a");
  const aChildren = c.parent.reduce((acc, p, i) => (p === aRow ? [...acc, i] : acc), []);
  assert.deepEqual(aChildren.map((i) => c.move[i]), ["a1", "a2", "…"]);
});

test("compressTree: expanding the root's bundle reveals all its children", () => {
  const c = compressTree(TREE, 4, new Set([""]));
  assert.deepEqual(c.move, ["", "a", "b", "c", "d", "e", "f", "a1", "a2", "a3"]);
  assert.ok(c.bundle.every((b) => b === null)); // "a" still fits 3 <= k=4
});

test("compressTree: expanding one node doesn't expand its siblings", () => {
  const c = compressTree(TREE, 2, new Set(["a"])); // only "a"'s fold opened
  // root still folds c,d,e,f behind "…"
  assert.deepEqual(c.move.slice(0, 4), ["", "a", "b", "…"]);
  // but "a"'s own children (a1,a2,a3) are all visible now
  const aRow = c.move.indexOf("a");
  const aChildren = c.parent.reduce((acc, p, i) => (p === aRow ? [...acc, i] : acc), []);
  assert.deepEqual(aChildren.map((i) => c.move[i]), ["a1", "a2", "a3"]);
});
