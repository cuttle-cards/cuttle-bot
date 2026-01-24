from fastapi.testclient import TestClient

from server.app import create_app
from server.session_store import SessionStore


class StubAI:
    async def get_action(self, game_state, actions):
        return actions[0]


def test_action_submission_triggers_ai_turn() -> None:
    store = SessionStore()
    app = create_app(session_store=store, ai_player_factory=StubAI)
    client = TestClient(app)

    create_response = client.post(
        "/api/sessions", json={"use_ai": True, "manual_selection": False}
    )
    assert create_response.status_code == 200
    payload = create_response.json()

    session_id = payload["session_id"]
    state_version = payload["state_version"]
    legal_actions = payload["legal_actions"]

    draw_action = next(
        (action for action in legal_actions if action["type"] == "Draw"), None
    )
    assert draw_action is not None

    action_response = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"state_version": state_version, "action_id": draw_action["id"]},
    )
    assert action_response.status_code == 200
    action_payload = action_response.json()

    assert action_payload["state_version"] > state_version
    last_actions = action_payload["last_actions"]
    assert len(last_actions) >= 2
    assert last_actions[0]["played_by"] == 0
    assert last_actions[1]["played_by"] == 1

    history_response = client.get(f"/api/sessions/{session_id}/history")
    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert len(history_payload["entries"]) >= 2


def test_stale_state_version_returns_409() -> None:
    store = SessionStore()
    app = create_app(session_store=store, ai_player_factory=StubAI)
    client = TestClient(app)

    create_response = client.post(
        "/api/sessions", json={"use_ai": True, "manual_selection": False}
    )
    payload = create_response.json()
    session_id = payload["session_id"]
    legal_actions = payload["legal_actions"]

    draw_action = next(
        (action for action in legal_actions if action["type"] == "Draw"), None
    )
    assert draw_action is not None

    response = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"state_version": payload["state_version"] + 1, "action_id": draw_action["id"]},
    )

    assert response.status_code == 409


def test_invalid_action_id_returns_400() -> None:
    store = SessionStore()
    app = create_app(session_store=store, ai_player_factory=StubAI)
    client = TestClient(app)

    create_response = client.post(
        "/api/sessions", json={"use_ai": True, "manual_selection": False}
    )
    payload = create_response.json()
    session_id = payload["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/actions",
        json={"state_version": payload["state_version"], "action_id": 999},
    )

    assert response.status_code == 400
