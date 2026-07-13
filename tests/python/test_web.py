"""Web server API tests: state/move/new round-trips, search-and-play flow,
and the WebSocket event stream."""

import time

import pytest
from fastapi.testclient import TestClient

from chessengine.engine import EngineConfig, SearchLimits
from chessengine.ui.web.server import Session, create_app

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def client():
    session = Session(
        config=EngineConfig(workers=1, seed=7),
        limits=SearchLimits(max_time_ms=3000, max_simulations=300),
    )
    with TestClient(create_app(session)) as c:
        yield c


def wait_for_idle(client, timeout_s=15.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = client.get("/api/state").json()
        if not state["searching"]:
            return state
        time.sleep(0.05)
    raise TimeoutError("search did not finish")


def test_initial_state(client):
    state = client.get("/api/state").json()
    assert state["fen"] == START_FEN
    assert state["turn"] == "w"
    assert len(state["legal_moves"]) == 20
    assert state["history"] == []
    assert state["outcome"] is None
    assert not state["searching"]


def test_move_roundtrip(client):
    state = client.post("/api/move", json={"uci": "e2e4"}).json()
    assert state["turn"] == "b"
    assert state["history"] == ["e4"]
    assert state["last_move"] == "e2e4"


def test_illegal_move_rejected(client):
    response = client.post("/api/move", json={"uci": "e2e5"})
    assert response.status_code == 400
    assert client.get("/api/state").json()["history"] == []


def test_new_game_resets(client):
    client.post("/api/move", json={"uci": "e2e4"})
    state = client.post("/api/new", json={}).json()
    assert state["fen"] == START_FEN
    assert state["history"] == []


def test_new_game_from_fen(client):
    fen = "8/8/8/8/8/5k2/6q1/7K w - - 0 1"
    state = client.post("/api/new", json={"fen": fen}).json()
    assert state["fen"] == fen
    assert client.post("/api/new", json={"fen": "not a fen"}).status_code == 400


def test_check_square_reported(client):
    # fool's mate pattern minus the mate: expose the white king to a check
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        client.post("/api/move", json={"uci": uci})
    state = client.get("/api/state").json()
    assert state["check_square"] == "e1"
    assert state["outcome"] == {"result": "0-1", "termination": "checkmate"}


def test_search_plays_move(client):
    assert client.post("/api/search/start").json()["searching"]
    state = wait_for_idle(client)
    assert len(state["history"]) == 1
    assert state["turn"] == "b"


def test_search_stop_plays_best_so_far(client):
    client.post("/api/search/start")
    time.sleep(0.3)  # let some simulations run so there is a best-so-far move
    state = client.post("/api/search/stop").json()
    assert not state["searching"]
    assert len(state["history"]) == 1


def test_move_rejected_while_searching():
    # own session with an effectively unbounded search: the tiny shared-fixture
    # search can finish before the move request is even handled
    session = Session(
        config=EngineConfig(workers=1),
        limits=SearchLimits(max_time_ms=30_000, convergence_window=0),
    )
    with TestClient(create_app(session)) as client:
        client.post("/api/search/start")
        assert client.post("/api/move", json={"uci": "e2e4"}).status_code == 409
        client.post("/api/search/stop")
        wait_for_idle(client)


def test_websocket_streams_state_and_search(client):
    with client.websocket_connect("/ws/events") as ws:
        first = ws.receive_json()
        assert first["type"] == "state"
        assert first["fen"] == START_FEN

        client.post("/api/search/start")
        seen = set()
        deadline = time.monotonic() + 15
        while "search_end" not in seen and time.monotonic() < deadline:
            event = ws.receive_json()
            seen.add(event["type"])
            if event["type"] == "search_end":
                assert event["played_move"]
                assert event["stop_reason"] in {"time", "converged", "simulations"}
        assert {"state", "stats", "search_end"} <= seen
        # the state broadcast right after search_end must already say idle,
        # or the client leaves the board locked (regression)
        event = ws.receive_json()
        assert event["type"] == "state"
        assert not event["searching"]
        assert len(event["history"]) == 1
        # final state after the engine's move
        state = wait_for_idle(client)
        assert len(state["history"]) == 1


def test_goto_ply_takeback(client):
    for uci in ["e2e4", "e7e5", "g1f3"]:
        client.post("/api/move", json={"uci": uci})
    state = client.post("/api/goto", json={"ply": 1}).json()
    assert state["history"] == ["e4"]
    assert state["turn"] == "b"
    assert client.post("/api/goto", json={"ply": 5}).status_code == 400
    # rewound game keeps playing fine (engine tree was re-rooted)
    state = client.post("/api/move", json={"uci": "e7e5"}).json()
    assert state["history"] == ["e4", "e5"]


def test_goto_path_explores_forward(client):
    state = client.post("/api/goto", json={"path": ["e2e4", "e7e5"]}).json()
    assert state["history"] == ["e4", "e5"]
    assert state["turn"] == "w"
    assert client.post("/api/goto", json={"path": ["e2e4"]}).status_code == 400
    assert client.post("/api/goto", json={}).status_code == 400
    assert client.post("/api/goto", json={"ply": 0, "path": []}).status_code == 400


def test_position_applies_valid_fen(client):
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    state = client.post("/api/position", json={"fen": fen}).json()
    assert state["fen"] == fen
    assert client.post("/api/position", json={"fen": "not a fen"}).status_code == 400
    # parseable but invalid: no black king
    response = client.post("/api/position", json={"fen": "8/8/8/8/8/8/8/4K3 w - - 0 1"})
    assert response.status_code == 400
    assert "king" in response.json()["detail"]


def test_config_roundtrip(client):
    config = client.get("/api/config").json()
    assert config["limits"]["max_time_ms"] == 3000
    assert config["structural"]["workers"] == 1
    assert config["cp_scale"] == 400.0

    config = client.put("/api/config", json={"limits": {"c_puct": 2.0}}).json()
    assert config["limits"]["c_puct"] == 2.0
    assert config["limits"]["max_time_ms"] == 3000  # untouched
    assert client.put("/api/config", json={"limits": {"nope": 1}}).status_code == 400


def test_config_structural_rebuilds_engine(client):
    client.post("/api/search/start")
    wait_for_idle(client)
    assert len(client.get("/api/tree").json()["parent"]) > 10
    config = client.put("/api/config", json={"structural": {"workers": 2}}).json()
    assert config["structural"]["workers"] == 2
    # rebuild dropped the tree: bare root at the current position
    assert client.get("/api/tree").json()["parent"] == [-1]
    # the rebuilt engine still searches and plays
    client.post("/api/search/start")
    state = wait_for_idle(client)
    assert len(state["history"]) == 2


def test_tree_detail_and_fens(client):
    client.post("/api/search/start")
    wait_for_idle(client)
    tree = client.get("/api/tree").json()
    assert tree["root_path"] == []
    # the engine played its move, so the root moved one ply down; ask for the
    # subtree of the current root's best child
    child_move = tree["move"][1]
    detail = client.post(
        "/api/tree/detail", json={"root_path": [child_move], "max_nodes": 50}
    ).json()
    assert detail["root_path"] == [child_move]
    assert detail["move"][0] == ""
    assert len(detail["parent"]) <= 50

    fens = client.post("/api/tree/fens", json={"paths": [[], [child_move], ["e2e5"]]}).json()
    assert fens["fens"][0] == client.get("/api/state").json()["fen"]
    assert fens["sans"][0] is None  # root: no move leads into it
    assert fens["fens"][1] is not None
    assert fens["sans"][1]  # SAN of the move into the node, for card labels
    assert fens["fens"][2] is None  # unplayable path -> null, not an error


def test_tree_endpoint(client):
    tree = client.get("/api/tree").json()
    assert tree["type"] == "tree"
    assert tree["turn"] == "w"
    assert tree["parent"] == [-1]  # fresh session: bare root
    client.post("/api/search/start")
    wait_for_idle(client)
    tree = client.get("/api/tree").json()
    assert len(tree["parent"]) > 10
    assert len(tree["parent"]) == len(tree["move"]) == len(tree["visits"])


def test_websocket_streams_tree(client):
    with client.websocket_connect("/ws/events") as ws:
        assert ws.receive_json()["type"] == "state"
        client.post("/api/search/start")
        tree_events = []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            event = ws.receive_json()
            if event["type"] == "tree":
                tree_events.append(event)
            if event["type"] == "search_end":
                break
        # at least the first-tick and the final snapshot
        assert len(tree_events) >= 2
        assert len(tree_events[-1]["parent"]) > 10
        wait_for_idle(client)


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>chessengine</title>" in response.text
    # stale-module guard: static files must always be revalidated
    assert response.headers["cache-control"] == "no-cache"
    assert client.get("/tree.js").headers["cache-control"] == "no-cache"
