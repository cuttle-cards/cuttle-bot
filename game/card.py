from __future__ import annotations

from enum import Enum
from typing import List, Optional


class Card:
    """
    A class that represents a card in the game.

    has suit and rank
    """

    def __init__(
        self,
        id: str,
        suit: Suit,
        rank: Rank,
        played_by: Optional[int] = None,
        purpose: Optional[Purpose] = None,
        attachments: Optional[List[Card]] = None,
    ):
        self.id = id
        self.suit = suit
        self.rank = rank
        self.played_by = played_by
        self.purpose = purpose
        self.attachments = attachments if attachments is not None else list()

    def __str__(self):
        jack_prefix = "[Jack]" * len(self.attachments) + " " if len(self.attachments) > 0 else ""
        stolen_prefix = "[Stolen from opponent] " if self.is_stolen() else ""
        return f"{stolen_prefix}{jack_prefix}{self.rank.value[0]} of {self.suit.value[0]}"

    def __repr__(self):
        return self.__str__()

    def clear_player_info(self):
        self.played_by = None
        self.purpose = None
    
    def is_point_card(self) -> bool:
        return self.point_value() <= Rank.TEN.value[1]

    def point_value(self) -> int:
        return self.rank.value[1]

    def suit_value(self) -> int:
        """Return the suit value for comparison (Clubs=0, Diamonds=1, Hearts=2, Spades=3)."""
        return self.suit.value[1]

    def is_face_card(self) -> bool:
        return (
            self.point_value() >= Rank.JACK.value[1]
            or self.point_value() == Rank.EIGHT.value[1]
        )

    def is_one_off(self) -> bool:
        """Check if the card can be played as a one-off effect."""
        return self.rank in [Rank.ACE, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]
    
    def is_stolen(self) -> bool:
        return len(self.attachments) % 2 == 1


class Suit(Enum):
    """
    An Enum class that represents a suit of a card.
    """

    CLUBS = ("Clubs", 0)
    DIAMONDS = ("Diamonds", 1)
    HEARTS = ("Hearts", 2)
    SPADES = ("Spades", 3)


class Rank(Enum):
    """
    An Enum class that represents a rank of a card.
    """

    ACE = ("Ace", 1)
    TWO = ("Two", 2)
    THREE = ("Three", 3)
    FOUR = ("Four", 4)
    FIVE = ("Five", 5)
    SIX = ("Six", 6)
    SEVEN = ("Seven", 7)
    EIGHT = ("Eight", 8)
    NINE = ("Nine", 9)
    TEN = ("Ten", 10)
    JACK = ("Jack", 11)
    QUEEN = ("Queen", 12)
    KING = ("King", 13)


class Purpose(Enum):
    """
    An Enum class that represents a purpose of a card.
    """

    POINTS = "Points"
    FACE_CARD = "Face Card"
    ONE_OFF = "One Off"
    COUNTER = "Counter"
    JACK = "Jack"
