from fastapi.testclient import TestClient

from server.app import create_app
from server.session_store import SessionStore


def test_history_order_and_delete_session() -> None:
    store = SessionStore()
    app = create_app(session_store=store)
    client = TestClient(app)

    create_response = client.post(
        "/api/sessions", json={"use_ai": False, "manual_selection": False}
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

    history_response = client.get(f"/api/sessions/{session_id}/history")
    assert history_response.status_code == 200
    history_payload = history_response.json()

    entries = history_payload["entries"]
    assert len(entries) >= 1
    assert entries[0]["action_type"] == "Draw"
    assert entries[0]["player"] == 0

    delete_response = client.delete(f"/api/sessions/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True}

    missing_response = client.get(f"/api/sessions/{session_id}")
    assert missing_response.status_code == 404
