"""In-memory session store for game sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Protocol
from uuid import uuid4

from game.game import Game
from game.game_state import GameState

class AIPlayerProtocol(Protocol):
    async def get_action(self, game_state: GameState, legal_actions: list) -> object: ...
    def get_action_sync(self, game_state: GameState, legal_actions: list) -> object: ...
    def choose_card_from_discard(self, discard_pile: list) -> object: ...
    def choose_two_cards_from_hand(self, hand: list) -> list: ...


try:
    from game.ai_player import AIPlayer as LLMPlayer
except ImportError:  # pragma: no cover - defensive for limited environments
    LLMPlayer = None  # type: ignore[assignment]

try:
    from game.rl_ai_player import RLAIPlayerWrapper as RLPlayer
except ImportError:  # pragma: no cover - defensive for limited environments
    RLPlayer = None  # type: ignore[assignment]


@dataclass
class GameSession:
    """Container for a single game session."""

    id: str
    game: Game
    ai_player: Optional[AIPlayerProtocol]
    created_at: datetime
    updated_at: datetime
    state_version: int
    status: str


class SessionStore:
    """Thread-safe in-memory store for game sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, GameSession] = {}
        self._lock: Optional[asyncio.Lock] = None

    async def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def create_session(
        self,
        *,
        use_ai: bool = True,
        manual_selection: bool = False,
        ai_player_factory: Optional[Callable[[], AIPlayerProtocol]] = None,
        ai_type: str = "rl",
    ) -> GameSession:
        """Create and store a new session."""
        lock = await self._get_lock()
        async with lock:
            session_id = uuid4().hex
            ai_player = None
            if use_ai:
                if ai_player_factory is not None:
                    ai_player = ai_player_factory()
                elif ai_type == "llm":
                    if LLMPlayer is None:
                        raise ValueError("LLM AI is not available")
                    ai_player = LLMPlayer()
                elif ai_type == "rl":
                    if RLPlayer is None:
                        raise ValueError("RL AI is not available")
                    ai_player = RLPlayer()
                else:
                    raise ValueError(f"Unknown ai_type: {ai_type}")
            game = Game(
                manual_selection=manual_selection,
                ai_player=ai_player,
                input_mode="api",
            )
            now = datetime.utcnow()
            session = GameSession(
                id=session_id,
                game=game,
                ai_player=ai_player,
                created_at=now,
                updated_at=now,
                state_version=0,
                status="active",
            )
            self._sessions[session_id] = session
            return session

    async def get_session(self, session_id: str) -> Optional[GameSession]:
        """Fetch a session by id."""
        lock = await self._get_lock()
        async with lock:
            return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by id."""
        lock = await self._get_lock()
        async with lock:
            return self._sessions.pop(session_id, None) is not None

    async def session_count(self) -> int:
        """Return number of active sessions."""
        lock = await self._get_lock()
        async with lock:
            return len(self._sessions)
