"""
Card module for the Cuttle card game.

This module defines the core card-related classes and enums used in the game:
- Card: The main class representing a playing card
- Suit: Enum for card suits (Clubs, Diamonds, Hearts, Spades)
- Rank: Enum for card ranks (Ace through King)
- Purpose: Enum for card purposes in the game
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class Card:
    """A class representing a playing card in the Cuttle game.

    Each card has a suit, rank, and various game-specific attributes such as
    who played it, its current purpose, and any cards attached to it (e.g., by Jacks).

    Attributes:
        id (str): Unique identifier for the card.
        suit (Suit): The card's suit (Clubs, Diamonds, Hearts, Spades).
        rank (Rank): The card's rank (Ace through King).
        played_by (Optional[int]): Index of the player who played this card (0 or 1).
        purpose (Optional[Purpose]): Current purpose of the card in the game.
        attachments (List[Card]): List of cards attached to this card (e.g., by Jacks).
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
        """Initialize a new Card instance.

        Args:
            id (str): Unique identifier for the card.
            suit (Suit): The card's suit.
            rank (Rank): The card's rank.
            played_by (Optional[int], optional): Player who played the card. Defaults to None.
            purpose (Optional[Purpose], optional): Card's purpose in the game. Defaults to None.
            attachments (Optional[List[Card]], optional): Attached cards. Defaults to None.
        """
        self.id = id
        self.suit = suit
        self.rank = rank
        self.played_by = played_by
        self.purpose = purpose
        self.attachments = attachments if attachments is not None else list()

    def __str__(self) -> str:
        """Get a string representation of the card.

        The string includes:
        - A [Jack] prefix for each Jack attached to the card
        - A [Stolen from opponent] prefix if the card is stolen
        - The card's rank and suit

        Returns:
            str: String representation of the card.
        """
        jack_prefix = (
            "[Jack]" * len(self.attachments) + " " if len(self.attachments) > 0 else ""
        )
        stolen_prefix = "[Stolen from opponent] " if self.is_stolen() else ""
        return (
            f"{stolen_prefix}{jack_prefix}{self.rank.value[0]} of {self.suit.value[0]}"
        )

    def __repr__(self) -> str:
        """Get a string representation of the card for debugging.

        Returns:
            str: Same as __str__() for simplicity.
        """
        return self.__str__()

    def clear_player_info(self) -> None:
        """Clear the card's player-specific information.

        Resets:
        - The player who played the card
        - The card's current purpose
        """
        self.played_by = None
        self.purpose = None

    def is_point_card(self) -> bool:
        """Check if the card can be played for points.

        A card can be played for points if its value is 10 or less.

        Returns:
            bool: True if the card can be played for points.
        """
        return self.point_value() <= Rank.TEN.value[1]

    def point_value(self) -> int:
        """Get the card's point value.

        The point value is the same as the rank's numeric value:
        - Ace = 1
        - Number cards = their number
        - Face cards = 11 (Jack), 12 (Queen), 13 (King)

        Returns:
            int: The card's point value.
        """
        return self.rank.value[1]

    def suit_value(self) -> int:
        """Get the card's suit value for comparison.

        Suit values are:
        - Clubs = 0
        - Diamonds = 1
        - Hearts = 2
        - Spades = 3

        Returns:
            int: The suit's numeric value.
        """
        return self.suit.value[1]

    def is_face_card(self) -> bool:
        """Check if the card is a face card.

        Face cards are:
        - Jack (11)
        - Queen (12)
        - King (13)
        - Eight (8) - special case in Cuttle

        Returns:
            bool: True if the card is a face card.
        """
        return (
            self.point_value() >= Rank.JACK.value[1]
            or self.point_value() == Rank.EIGHT.value[1]
        )

    def is_one_off(self) -> bool:
        """Check if the card can be played as a one-off effect.

        One-off cards are:
        - Ace: Return target card to owner's hand
        - Three: Draw two cards
        - Four: Protection from targeted effects
        - Five: Draw a card from opponent's hand
        - Six: Opponent discards two cards

        Returns:
            bool: True if the card can be played as a one-off.
        """
        return self.rank in [
            Rank.ACE,
            Rank.THREE,
            Rank.FOUR,
            Rank.FIVE,
            Rank.SIX,
            Rank.SEVEN,
        ]

    def is_stolen(self) -> bool:
        """Check if the card is currently stolen by the opponent.

        A card is considered stolen if it has an odd number of Jacks
        attached to it, as each Jack switches control of the card.

        Returns:
            bool: True if the card is stolen.
        """
        return len(self.attachments) % 2 == 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Card object to a dictionary."""
        return {
            "id": self.id,
            "suit": self.suit.name,
            "rank": self.rank.name,
            "played_by": self.played_by,
            "purpose": self.purpose.name if self.purpose else None,
            "attachments": [att.to_dict() for att in self.attachments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Card:
        """Deserialize a Card object from a dictionary."""
        attachments_data = data.get("attachments", [])
        attachments = [
            cls.from_dict(att_data) for att_data in attachments_data if att_data
        ]
        purpose_name = data.get("purpose")
        return cls(
            id=data["id"],
            suit=Suit[data["suit"]],
            rank=Rank[data["rank"]],
            played_by=data.get("played_by"),
            purpose=Purpose[purpose_name] if purpose_name else None,
            attachments=attachments,
        )


class Suit(Enum):
    """Enumeration of card suits in the game.

    Each suit has:
    - A display name (e.g., "Clubs")
    - A numeric value for comparison (0-3)

    The suit order (lowest to highest) is:
    Clubs (0) < Diamonds (1) < Hearts (2) < Spades (3)
    """

    CLUBS = ("Clubs", 0)
    DIAMONDS = ("Diamonds", 1)
    HEARTS = ("Hearts", 2)
    SPADES = ("Spades", 3)


class Rank(Enum):
    """Enumeration of card ranks in the game.

    Each rank has:
    - A display name (e.g., "Ace")
    - A numeric value (1-13)

    Special ranks and their effects:
    - Ace (1): Return target card to owner's hand
    - Two (2): Scuttle point cards of equal or lesser value
    - Three (3): Draw two cards
    - Four (4): Protection from targeted effects
    - Five (5): Draw a card from opponent's hand
    - Six (6): Opponent discards two cards
    - Seven (7): Point card
    - Eight (8): Counts as a face card
    - Nine-Ten (9-10): Point cards
    - Jack (11): Steal opponent's card
    - Queen (12): Destroy target face card
    - King (13): Reduce points needed to win
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
    """Enumeration of possible card purposes in the game.

    A card's purpose indicates how it's currently being used:
    - POINTS: Card played for its point value
    - FACE_CARD: Card played for its face card effect
    - ONE_OFF: Card played for its one-off effect
    - COUNTER: Card used to counter another card
    - JACK: Card used as a Jack (stealing another card)
    """

    POINTS = "Points"
    FACE_CARD = "Face Card"
    ONE_OFF = "One Off"
    COUNTER = "Counter"
    JACK = "Jack"
    SCUTTLE = "Scuttle"
