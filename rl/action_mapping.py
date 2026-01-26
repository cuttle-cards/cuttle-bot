"""Action mapping for Cuttle RL.

Maps fixed action indices to game Action objects using card identity (rank/suit).
This avoids per-turn reindexing based on legal action list order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from game.action import Action, ActionType
from game.card import Card


NUM_CARDS = 52
PAIR_SIZE = NUM_CARDS * NUM_CARDS


@dataclass(frozen=True)
class ActionGroup:
    name: str
    size: int


ACTION_GROUPS = (
    ActionGroup("draw", 1),
    ActionGroup("resolve", 1),
    ActionGroup("points", NUM_CARDS),
    ActionGroup("face", NUM_CARDS),
    ActionGroup("one_off", NUM_CARDS),
    ActionGroup("one_off_target", PAIR_SIZE),
    ActionGroup("counter", NUM_CARDS),
    ActionGroup("take_from_discard", NUM_CARDS),
    ActionGroup("discard_from_hand", NUM_CARDS),
    ActionGroup("discard_revealed", NUM_CARDS),
    ActionGroup("scuttle", PAIR_SIZE),
    ActionGroup("jack", PAIR_SIZE),
)


_OFFSETS: Dict[str, int] = {}
_running = 0
for _group in ACTION_GROUPS:
    _OFFSETS[_group.name] = _running
    _running += _group.size

ACTION_SPACE_SIZE = _running


def card_index(card: Card) -> int:
    """Return canonical 0..51 index for a card based on rank/suit."""
    return (card.rank.value[1] - 1) * 4 + card.suit.value[1]


def _pair_index(attacker_idx: int, target_idx: int) -> int:
    return attacker_idx * NUM_CARDS + target_idx


def action_to_index(action: Action) -> Optional[int]:
    """Map a concrete Action to a fixed action index."""
    if action.action_type == ActionType.DRAW:
        return _OFFSETS["draw"]
    if action.action_type == ActionType.RESOLVE:
        return _OFFSETS["resolve"]
    if action.action_type == ActionType.POINTS:
        if action.card is None:
            return None
        return _OFFSETS["points"] + card_index(action.card)
    if action.action_type == ActionType.FACE_CARD:
        if action.card is None:
            return None
        return _OFFSETS["face"] + card_index(action.card)
    if action.action_type == ActionType.ONE_OFF:
        if action.card is None:
            return None
        attacker_idx = card_index(action.card)
        if action.target is None:
            return _OFFSETS["one_off"] + attacker_idx
        target_idx = card_index(action.target)
        return _OFFSETS["one_off_target"] + _pair_index(attacker_idx, target_idx)
    if action.action_type == ActionType.COUNTER:
        if action.card is None:
            return None
        return _OFFSETS["counter"] + card_index(action.card)
    if action.action_type == ActionType.TAKE_FROM_DISCARD:
        if action.card is None:
            return None
        return _OFFSETS["take_from_discard"] + card_index(action.card)
    if action.action_type == ActionType.DISCARD_FROM_HAND:
        if action.card is None:
            return None
        return _OFFSETS["discard_from_hand"] + card_index(action.card)
    if action.action_type == ActionType.DISCARD_REVEALED:
        if action.card is None:
            return None
        return _OFFSETS["discard_revealed"] + card_index(action.card)
    if action.action_type == ActionType.SCUTTLE:
        if action.card is None or action.target is None:
            return None
        attacker_idx = card_index(action.card)
        target_idx = card_index(action.target)
        return _OFFSETS["scuttle"] + _pair_index(attacker_idx, target_idx)
    if action.action_type == ActionType.JACK:
        if action.card is None or action.target is None:
            return None
        attacker_idx = card_index(action.card)
        target_idx = card_index(action.target)
        return _OFFSETS["jack"] + _pair_index(attacker_idx, target_idx)
    return None


def build_action_map(legal_actions: Iterable[Action]) -> Dict[int, Action]:
    """Build a mapping from fixed action index to Action for the current state."""
    index_to_action: Dict[int, Action] = {}
    for action in legal_actions:
        idx = action_to_index(action)
        if idx is None:
            continue
        if idx in index_to_action:
            continue
        index_to_action[idx] = action
    return index_to_action


def legal_action_mask_from_actions(
    legal_actions: Iterable[Action],
    action_space_size: int = ACTION_SPACE_SIZE,
) -> np.ndarray:
    """Return a boolean mask over the full action space for given legal actions."""
    mask = np.zeros(action_space_size, dtype=np.bool_)
    for idx in build_action_map(legal_actions).keys():
        if 0 <= idx < action_space_size:
            mask[idx] = True
    return mask


def legal_action_mask(game_state) -> np.ndarray:
    """Return a boolean mask over the full action space for a game state."""
    return legal_action_mask_from_actions(
        game_state.get_legal_actions(),
        action_space_size=ACTION_SPACE_SIZE,
    )


def action_index_to_action(game_state, action_index: int) -> Optional[Action]:
    """Resolve a fixed action index into a concrete legal Action, if any."""
    action_map = build_action_map(game_state.get_legal_actions())
    return action_map.get(action_index)
