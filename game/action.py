"""
Action module for the Cuttle card game.

This module defines the classes and enums related to game actions:
- Action: The main class representing a player's action
- ActionType: Enum for different types of actions
- ActionSource: Enum for where cards come from in actions
"""

from __future__ import annotations

from game.card import Card
from enum import Enum


class ActionSource(Enum):
    """Enumeration of possible sources for cards in actions.

    This enum indicates where a card comes from when it's used in an action:
    - HAND: Card played from a player's hand
    - DECK: Card drawn from the deck
    - FIELD: Card on a player's field
    - DISCARD: Card from the discard pile
    """

    HAND = "Hand"
    DECK = "Deck"
    FIELD = "Field"
    DISCARD = "Discard"


class Action:
    """A class representing a player's action in the game.

    This class encapsulates all information about an action:
    - What type of action it is
    - Which card is being played
    - What the target is (if any)
    - Who is playing it
    - Where the card comes from
    - Whether it needs more input

    Attributes:
        action_type (ActionType): The type of action being performed.
        card (Card): The card being played or used.
        target (Card): The target card (if any) for the action.
        played_by (int): Index of the player performing the action (0 or 1).
        requires_additional_input (bool): Whether more input is needed.
        source (ActionSource): Where the card comes from.
    """

    action_type: ActionType
    card: Card
    target: Card
    played_by: int
    requires_additional_input: bool
    source: ActionSource

    def __init__(
        self,
        action_type: ActionType,
        card: Card,
        target: Card,
        played_by: int,
        requires_additional_input: bool = False,
        source: ActionSource = ActionSource.HAND,
    ):
        """Initialize a new Action instance.

        Args:
            action_type (ActionType): The type of action being performed.
            card (Card): The card being played or used.
            target (Card): The target card (if any) for the action.
            played_by (int): Index of the player performing the action (0 or 1).
            requires_additional_input (bool, optional): Whether more input is needed.
                Defaults to False.
            source (ActionSource, optional): Where the card comes from.
                Defaults to ActionSource.HAND.
        """
        self.action_type = action_type
        self.card = card
        self.target = target
        self.requires_additional_input = requires_additional_input
        self.played_by = played_by
        self.source = source

    def __repr__(self) -> str:
        """Get a string representation of the action.

        The representation varies based on the action type:
        - POINTS: "Play {card} as points"
        - FACE_CARD: "Play {card} as face card"
        - ONE_OFF: "Play {card} as one-off"
        - SCUTTLE: "Scuttle {target} on P{player}'s field with {card}"
        - DRAW: "Draw a card from deck"
        - COUNTER: "Counter {target} with {card}"
        - JACK: "Play {card} as jack on {target}"
        - RESOLVE: "Resolve one-off {target}"

        Returns:
            str: Human-readable description of the action.
        """
        if self.action_type == ActionType.POINTS:
            return f"Play {self.card} as points"
        elif self.action_type == ActionType.FACE_CARD:
            return f"Play {self.card} as face card"
        elif self.action_type == ActionType.ONE_OFF:
            return f"Play {self.card} as one-off"
        elif self.action_type == ActionType.SCUTTLE:
            return f"Scuttle {self.target} on P{self.target.played_by}'s field with {self.card}"
        elif self.action_type == ActionType.DRAW:
            return "Draw a card from deck"
        elif self.action_type == ActionType.COUNTER:
            return f"Counter {self.target} with {self.card}"
        elif self.action_type == ActionType.JACK:
            return f"Play {self.card} as jack on {self.target}"
        elif self.action_type == ActionType.RESOLVE:
            return f"Resolve one-off {self.target}"
        else:
            # Handle any unexpected action types
            return f"Unknown Action: {self.action_type.value} with card {self.card}"

    def __str__(self) -> str:
        """Get a string representation of the action.

        Returns:
            str: Same as __repr__() for simplicity.
        """
        return self.__repr__()


class ActionType(Enum):
    """Enumeration of possible action types in the game.

    This enum defines all possible actions a player can take:

    Basic Actions:
    - DRAW: Draw a card from the deck
    - POINTS: Play a card for points
    - FACE_CARD: Play a card for its face card effect
    - ONE_OFF: Play a card for its one-off effect
    - SCUTTLE: Use a card to destroy an opponent's point card
    - JACK: Use a Jack to steal an opponent's card

    Special Actions:
    - COUNTER: Counter another player's action
    - RESOLVE: Resolve a one-off effect

    Game State Actions:
    - REQUEST_STALEMATE: Ask for a stalemate
    - ACCEPT_STALEMATE: Accept a stalemate request
    - REJECT_STALEMATE: Reject a stalemate request
    - CONCEDE: Give up the game
    """

    DRAW = "Draw"
    POINTS = "Points"
    FACE_CARD = "Face Card"
    ONE_OFF = "One-Off"
    COUNTER = "Counter"
    RESOLVE = "Resolve"
    SCUTTLE = "Scuttle"
    JACK = "Jack"
    REQUEST_STALEMATE = "Request Stalemate"
    ACCEPT_STALEMATE = "Accept Stalemate"
    REJECT_STALEMATE = "Reject Stalemate"
    CONCEDE = "Concede"
