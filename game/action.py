from __future__ import annotations

from game.card import Card
from enum import Enum


class Action:
    """
    A class that represents an action in the game.
    """

    action_type: ActionType
    card: Card
    target: Card
    played_by: int
    requires_additional_input: bool

    def __init__(
        self,
        action_type: ActionType,
        card: Card,
        target: Card,
        played_by: int,
        requires_additional_input: bool = False,
    ):
        self.action_type = action_type
        self.card = card
        self.target = target
        self.requires_additional_input = requires_additional_input
        self.played_by = played_by

    def __repr__(self):

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
        elif self.action_type == ActionType.RESOLVE:
            return f"Resolve one-off {self.target}"
        elif self.action_type == ActionType.END_GAME:
            return "End game"

    def __str__(self):
        return self.__repr__()


class ActionType(Enum):
    """
    An Enum class that represents the type of an action.
    """

    DRAW = "Draw"
    POINTS = "Points"
    FACE_CARD = "Face Card"
    ONE_OFF = "One-Off"
    COUNTER = "Counter"
    RESOLVE = "Resolve"
    SCUTTLE = "Scuttle"
    REQUEST_STALEMATE = "Request Stalemate"
    ACCEPT_STALEMATE = "Accept Stalemate"
    REJECT_STALEMATE = "Reject Stalemate"
    CONCEDE = "Concede"
    END_GAME = "End Game"
