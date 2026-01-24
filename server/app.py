"""FastAPI application entrypoint."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles

from game.action import Action
from game.game import Game
from server.models import ActionRequest, CreateSessionRequest
from server.session_store import GameSession, SessionStore
from server.views import action_view, actions_view, game_state_view

from server.session_store import AIPlayerProtocol


def _update_game_state(game: Game, turn_finished: bool) -> None:
    if turn_finished:
        game.game_state.resolving_one_off = False

    if (
        game.game_state.resolving_three
        or game.game_state.resolving_four
        or game.game_state.resolving_seven
    ):
        return

    if game.game_state.resolving_one_off:
        game.game_state.next_player()
    else:
        game.game_state.next_turn()


def _is_ai_turn(game: Game) -> bool:
    state = game.game_state
    return (
        state.use_ai
        and (
            (state.resolving_one_off and state.current_action_player == 1)
            or (state.resolving_four and state.current_action_player == 1)
            or (state.resolving_seven and state.current_action_player == 1)
            or (not state.resolving_one_off and state.turn == 1)
        )
    )


async def _apply_action(session: GameSession, action: Action) -> None:
    turn_finished, should_stop, _winner = session.game.game_state.update_state(action)
    if not should_stop:
        _update_game_state(session.game, turn_finished)
    if should_stop:
        session.status = "ended"
    session.updated_at = datetime.utcnow()
    session.state_version += 1


async def _apply_ai_turns(session: GameSession) -> List[Action]:
    if session.ai_player is None:
        return []

    applied: List[Action] = []
    while _is_ai_turn(session.game):
        legal_actions = session.game.game_state.get_legal_actions()
        if not legal_actions:
            break
        try:
            chosen_action = await session.ai_player.get_action(
                session.game.game_state, legal_actions
            )
        except Exception:
            chosen_action = legal_actions[0]

        applied.append(chosen_action)
        await _apply_action(session, chosen_action)

        if session.status == "ended":
            break

    return applied


def create_app(
    session_store: Optional[SessionStore] = None,
    ai_player_factory: Optional[Callable[[], AIPlayerProtocol]] = None,
) -> FastAPI:
    """Create and configure the FastAPI app."""
    store = session_store or SessionStore()
    app = FastAPI(title="Cuttle API")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/sessions")
    async def create_session(payload: CreateSessionRequest) -> dict:
        try:
            session = await store.create_session(
                use_ai=payload.use_ai,
                manual_selection=payload.manual_selection,
                ai_player_factory=ai_player_factory,
                ai_type=payload.ai_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        hide_hand = 1 if payload.use_ai else None
        legal_actions = session.game.game_state.get_legal_actions()
        return {
            "session_id": session.id,
            "state": game_state_view(session.game.game_state, hide_player_hand=hide_hand),
            "legal_actions": actions_view(legal_actions),
            "state_version": session.state_version,
            "ai_thinking": False,
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        hide_hand = 1 if session.game.game_state.use_ai else None
        legal_actions = session.game.game_state.get_legal_actions()
        return {
            "session_id": session.id,
            "state": game_state_view(session.game.game_state, hide_player_hand=hide_hand),
            "legal_actions": actions_view(legal_actions),
            "state_version": session.state_version,
            "ai_thinking": False,
        }

    @app.get("/api/sessions/{session_id}/actions")
    async def get_actions(session_id: str) -> dict:
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        legal_actions = session.game.game_state.get_legal_actions()
        return {
            "state_version": session.state_version,
            "legal_actions": actions_view(legal_actions),
        }

    @app.post("/api/sessions/{session_id}/actions")
    async def submit_action(session_id: str, payload: ActionRequest) -> dict:
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if payload.state_version != session.state_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="State version mismatch",
            )
        legal_actions = session.game.game_state.get_legal_actions()
        if not legal_actions:
            raise HTTPException(status_code=400, detail="No legal actions available")
        if payload.action_id < 0 or payload.action_id >= len(legal_actions):
            raise HTTPException(status_code=400, detail="Invalid action id")

        chosen_action = legal_actions[payload.action_id]
        applied_actions: List[Action] = [chosen_action]
        await _apply_action(session, chosen_action)

        if session.status != "ended":
            applied_actions.extend(await _apply_ai_turns(session))

        hide_hand = 1 if session.game.game_state.use_ai else None
        updated_actions = session.game.game_state.get_legal_actions()
        return {
            "state": game_state_view(session.game.game_state, hide_player_hand=hide_hand),
            "legal_actions": actions_view(updated_actions),
            "state_version": session.state_version,
            "last_actions": [
                action_view(action, action_id=-1) for action in applied_actions
            ],
        }

    @app.get("/api/sessions/{session_id}/history")
    async def get_history(session_id: str) -> dict:
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.game.game_state.game_history.to_dict()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        deleted = await store.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}

    dist_dir = Path(__file__).resolve().parents[1] / "web" / "dist"
    if dist_dir.is_dir():
        # Serve built frontend assets from the same FastAPI process.
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
