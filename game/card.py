from __future__ import annotations

from enum import Enum
from typing import List, Optional

class Card:
    """
    A class that represents a card in the game.

    has suit and rank
    """
    def __init__(self, id: str, suit: Suit, rank: Rank, attachments: Optional[List[Card]]=[], played_by: Optional[int] = None, purpose: Optional[Purpose] = None):
        self.id = id
        self.suit = suit
        self.rank = rank
        self.attachments = attachments
        self.played_by = played_by
        self.purpose = purpose
    
    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def __repr__(self):
        return self.__str__()

    def clear_player_info(self):
        self.played_by = None
        self.purpose = None



class Suit(Enum):
    """
    An Enum class that represents a suit of a card.
    """
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

class Rank(Enum):
    """
    An Enum class that represents a rank of a card.
    """
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13

class Purpose(Enum):
    """
    An Enum class that represents a purpose of a card.
    """
    POINTS = 0
    FACE_CARD = 1
    ONE_OFF = 2
