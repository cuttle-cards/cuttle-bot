"""Serialization module for the Cuttle card game.

This module provides functionality for serializing and deserializing game objects,
including cards and game states. It supports saving and loading game states to/from
JSON files, enabling game persistence and state transfer between different instances.
"""

from __future__ import annotations
import json
from typing import Dict, List, Tuple
from game.card import Card, Suit, Rank, Purpose
from game.game_state import GameState


def serialize_card(card: Card) -> Dict:
    """Serialize a Card object to a dictionary representation.

    This function converts a Card object into a dictionary containing all its
    relevant attributes, making it suitable for JSON serialization.

    Args:
        card: The Card object to serialize.

    Returns:
        Dict: A dictionary containing the card's attributes:
            - id: The card's unique identifier
            - suit: The card's suit name
            - rank: The card's rank name
            - played_by: The player who played the card
            - purpose: The card's purpose name (if any)
    """
    return {
        "id": str(card.id),
        "suit": card.suit.name,
        "rank": card.rank.name,
        "played_by": card.played_by,
        "purpose": card.purpose.name if card.purpose else None,
    }


def deserialize_card(data: Dict) -> Card:
    """Deserialize a dictionary into a Card object.

    This function reconstructs a Card object from its dictionary representation,
    restoring all its attributes including suit, rank, and purpose.

    Args:
        data: Dictionary containing the card's serialized attributes.

    Returns:
        Card: A new Card object with the deserialized attributes.

    Raises:
        KeyError: If required attributes are missing from the input dictionary.
    """
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
    """Serialize a GameState object to a dictionary representation.

    This function converts a GameState object into a dictionary containing all its
    components, including hands, fields, deck, and game status information.

    Args:
        game_state: The GameState object to serialize.

    Returns:
        Dict: A dictionary containing:
            - hands: Serialized cards in each player's hand
            - fields: Serialized cards in each player's field
            - deck: Serialized cards in the deck
            - discard_pile: Serialized cards in the discard pile
            - turn: Current turn number
            - last_action_played_by: Player who played the last action
            - current_action_player: Current player's turn
            - status: Current game status
            - resolving_two: Whether a two is being resolved
            - resolving_one_off: Whether a one-off is being resolved
    """
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
    """Deserialize a dictionary into a GameState object.

    This function reconstructs a GameState object from its dictionary representation,
    restoring all game components and state information.

    Args:
        data: Dictionary containing the serialized game state.

    Returns:
        GameState: A new GameState object with the deserialized components.

    Raises:
        KeyError: If required components are missing from the input dictionary.
    """
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


def save_game_state(game_state: GameState, filename: str) -> None:
    """Save a game state to a JSON file.

    This function serializes a GameState object and writes it to a JSON file,
    enabling game state persistence.

    Args:
        game_state: The GameState object to save.
        filename: Path to the file where the game state will be saved.

    Raises:
        IOError: If the file cannot be written.
    """
    data = serialize_game_state(game_state)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def load_game_state(filename: str) -> GameState:
    """Load a game state from a JSON file.

    This function reads a JSON file containing a serialized game state and
    reconstructs the GameState object.

    Args:
        filename: Path to the file containing the saved game state.

    Returns:
        GameState: The reconstructed GameState object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If the JSON data is missing required components.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return deserialize_game_state(data)
