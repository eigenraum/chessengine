"""Self-play data generation and the training loop (DESIGN-M6.md section 7).

Pure Python; torch stays behind lazy imports inside the functions that
actually need it, so shard I/O and row-filtering logic (dataset.py) and the
CLI plumbing in selfplay.py/train.py/arena.py stay testable without the
`train` dependency group installed.

A generation is: self-play -> train -> arena -> promote.

    chessengine-selfplay --net best.pt --out data/gen3 --games 500 --jobs 8
    chessengine-train --data data --in best.pt --out candidate.pt
    chessengine-arena --net-a candidate.pt --net-b best.pt
    # promoted? -> cp candidate.pt best.pt

Generation 0's "current best" is a random-initialized net:

    uv run --group train python -c \
        "from chessengine.eval.torch_eval import TorchEvaluator; TorchEvaluator().save('best.pt')"

(equivalently: `chessengine-train --init --out best.pt`).

Real (non-CI) acceptance run: generation 1's net must beat the random-init
net at >= 55% over 100 arena games (`chessengine-arena --gate 0.55`), and
both loss components printed by `chessengine-train` must decrease between
generation 0 and generation 1 data. The tests in test_training.py are a
plumbing smoke gate (tiny nets/games/steps), not a claim about strength.
"""
