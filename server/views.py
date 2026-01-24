"""Serialization helpers for game state and actions."""

from __future__ import annotations

from typing import List, Optional

from game.action import Action
from game.card import Card
from game.game_state import GameState
from server.models import ActionView, CardView, GameStateView


def card_view(card: Card) -> CardView:
    """Create a CardView for API responses."""
    return CardView(
        id=card.id,
        suit=card.suit.name,
        rank=card.rank.name,
        display=str(card),
        played_by=card.played_by,
        purpose=card.purpose.name if card.purpose else None,
        point_value=card.point_value(),
        is_stolen=card.is_stolen(),
        attachments=[card_view(att) for att in card.attachments],
    )


def action_view(action: Action, action_id: int) -> ActionView:
    """Create an ActionView for API responses."""
    return ActionView(
        id=action_id,
        label=str(action),
        type=action.action_type.value,
        played_by=action.played_by,
        source=action.source.value,
        requires_additional_input=action.requires_additional_input,
        card=card_view(action.card) if action.card else None,
        target=card_view(action.target) if action.target else None,
    )


def actions_view(actions: List[Action]) -> List[ActionView]:
    """Create ActionViews for a list of actions."""
    return [action_view(action, action_id=i) for i, action in enumerate(actions)]


def game_state_view(
    game_state: GameState, *, hide_player_hand: Optional[int] = None
) -> GameStateView:
    """Create a GameStateView for API responses."""
    hands: List[List[CardView]] = []
    hand_counts: List[int] = []
    for idx, hand in enumerate(game_state.hands):
        hand_counts.append(len(hand))
        if hide_player_hand is not None and idx == hide_player_hand:
            hands.append([])
        else:
            hands.append([card_view(card) for card in hand])

    fields = [[card_view(card) for card in field] for field in game_state.fields]
    effective_fields = [
        [card_view(card) for card in game_state.get_player_field(player)]
        for player in range(len(game_state.hands))
    ]
    discard = [card_view(card) for card in game_state.discard_pile]
    scores = [
        game_state.get_player_score(player) for player in range(len(game_state.hands))
    ]
    targets = [
        game_state.get_player_target(player) for player in range(len(game_state.hands))
    ]

    return GameStateView(
        hands=hands,
        hand_counts=hand_counts,
        fields=fields,
        effective_fields=effective_fields,
        deck_count=len(game_state.deck),
        discard_pile=discard,
        discard_count=len(game_state.discard_pile),
        scores=scores,
        targets=targets,
        turn=game_state.turn,
        current_action_player=game_state.current_action_player,
        status=game_state.status,
        resolving_two=game_state.resolving_two,
        resolving_one_off=game_state.resolving_one_off,
        resolving_three=game_state.resolving_three,
        resolving_seven=game_state.resolving_seven,
        pending_seven_player=game_state.pending_seven_player,
        pending_seven_cards=[card_view(card) for card in game_state.pending_seven_cards],
        pending_seven_requires_discard=game_state.pending_seven_requires_discard,
        resolving_four=game_state.resolving_four,
        overall_turn=game_state.overall_turn,
        use_ai=game_state.use_ai,
        one_off_card_to_counter=card_view(game_state.one_off_card_to_counter)
        if game_state.one_off_card_to_counter
        else None,
        pending_four_count=game_state.pending_four_count,
    )
