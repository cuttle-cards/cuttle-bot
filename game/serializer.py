from __future__ import annotations
import json
from typing import Dict, List, Tuple
from game.card import Card, Suit, Rank, Purpose
from game.game_state import GameState


def serialize_card(card: Card) -> Dict:
    """Serialize a card to a dictionary."""
    return {
        "id": str(card.id),
        "suit": card.suit.name,
        "rank": card.rank.name,
        "played_by": card.played_by,
        "purpose": card.purpose.name if card.purpose else None,
    }


def deserialize_card(data: Dict) -> Card:
    """Deserialize a card from a dictionary."""
    card = Card(
        id=data["id"],
        suit=Suit[data["suit"]],
        rank=Rank[data["rank"]],
        played_by=data["played_by"],
    )
    if data["purpose"]:
        card.purpose = Purpose[data["purpose"]]
    return card


def serialize_game_state(game_state: GameState) -> Dict:
    """Serialize a game state to a dictionary."""
    return {
        "hands": [[serialize_card(card) for card in hand] for hand in game_state.hands],
        "fields": [
            [serialize_card(card) for card in field] for field in game_state.fields
        ],
        "deck": [serialize_card(card) for card in game_state.deck],
        "discard_pile": [serialize_card(card) for card in game_state.discard_pile],
        "turn": game_state.turn,
        "last_action_played_by": game_state.last_action_played_by,
        "current_action_player": game_state.current_action_player,
        "status": game_state.status,
        "resolving_two": game_state.resolving_two,
        "resolving_one_off": game_state.resolving_one_off,
    }


def deserialize_game_state(data: Dict) -> GameState:
    """Deserialize a game state from a dictionary."""
    hands = [
        [deserialize_card(card_data) for card_data in hand] for hand in data["hands"]
    ]
    fields = [
        [deserialize_card(card_data) for card_data in field] for field in data["fields"]
    ]
    deck = [deserialize_card(card_data) for card_data in data["deck"]]
    discard_pile = [deserialize_card(card_data) for card_data in data["discard_pile"]]

    if not fields:
        fields = [[], []]

    game_state = GameState(hands, fields, deck, discard_pile)
    game_state.turn = data["turn"]
    game_state.last_action_played_by = data["last_action_played_by"]
    game_state.current_action_player = data["current_action_player"]
    game_state.status = data["status"]
    game_state.resolving_two = data["resolving_two"]
    game_state.resolving_one_off = data["resolving_one_off"]

    return game_state


def save_game_state(game_state: GameState, filename: str):
    """Save a game state to a file."""
    data = serialize_game_state(game_state)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_game_state(filename: str) -> GameState:
    """Load a game state from a file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return deserialize_game_state(data)
