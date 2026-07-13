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


def test_move_rejected_while_searching(client):
    client.post("/api/search/start")
    assert client.post("/api/move", json={"uci": "e2e4"}).status_code == 409
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
        # final state after the engine's move
        state = wait_for_idle(client)
        assert len(state["history"]) == 1


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>chessengine</title>" in response.text
