"""Pydantic models for API responses."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class CardView(BaseModel):
    """Serializable view of a card for the UI."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    suit: str
    rank: str
    display: str
    played_by: Optional[int]
    purpose: Optional[str]
    point_value: int
    is_stolen: bool
    attachments: List["CardView"]


class ActionView(BaseModel):
    """Serializable view of an action for the UI."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    type: str
    played_by: int
    source: str
    requires_additional_input: bool
    card: Optional[CardView]
    target: Optional[CardView]


class GameStateView(BaseModel):
    """Serializable view of the game state for the UI."""

    model_config = ConfigDict(from_attributes=True)

    hands: List[List[CardView]]
    hand_counts: List[int]
    fields: List[List[CardView]]
    effective_fields: List[List[CardView]]
    deck_count: int
    discard_pile: List[CardView]
    discard_count: int
    scores: List[int]
    targets: List[int]
    turn: int
    current_action_player: int
    status: Optional[str]
    resolving_two: bool
    resolving_one_off: bool
    resolving_three: bool
    resolving_seven: bool
    pending_seven_player: Optional[int]
    pending_seven_cards: List[CardView]
    pending_seven_requires_discard: bool
    resolving_four: bool
    overall_turn: int
    use_ai: bool
    one_off_card_to_counter: Optional[CardView]
    pending_four_count: int


class CreateSessionRequest(BaseModel):
    """Request payload for creating a session."""

    use_ai: bool = True
    manual_selection: bool = False
    ai_type: Literal["llm", "rl"] = "rl"


class ActionRequest(BaseModel):
    """Request payload for submitting a player action."""

    state_version: int
    action_id: int
