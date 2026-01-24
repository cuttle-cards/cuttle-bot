"""
Game history module for the Cuttle card game.

This module provides the GameHistory and GameHistoryEntry classes to track
all meaningful game actions chronologically. This replaces log-based testing
with structured, queryable game action tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from game.action import ActionType
from game.card import Card


@dataclass
class GameHistoryEntry:
    """Represents a single action in the game history.
    
    This class captures all relevant information about a game action:
    - When it happened (timestamp, turn_number)
    - Who performed it (player)
    - What action was taken (action_type)
    - What cards were involved (card, target)
    - Where cards moved (source_location, destination_location)
    - Additional context (additional_data, description)
    
    Attributes:
        timestamp (datetime): When the action occurred.
        turn_number (int): The turn number when this action happened.
        player (int): The player who performed the action (0 or 1).
        action_type (ActionType): The type of action performed.
        card (Optional[Card]): The primary card involved in the action.
        target (Optional[Card]): The target card (if any) for the action.
        source_location (str): Where the card came from ("hand", "deck", "field", "discard_pile").
        destination_location (str): Where the card went ("hand", "field", "discard_pile").
        additional_data (Dict[str, Any]): Additional context data for special cases.
        description (str): Human-readable description of the action.
    """
    
    timestamp: datetime
    turn_number: int
    player: int
    action_type: ActionType
    card: Optional[Card] = None
    target: Optional[Card] = None
    source_location: str = ""
    destination_location: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entry.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "player": self.player,
            "action_type": self.action_type.value,
            "card": self.card.to_dict() if self.card else None,
            "target": self.target.to_dict() if self.target else None,
            "source_location": self.source_location,
            "destination_location": self.destination_location,
            "additional_data": self.additional_data,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameHistoryEntry":
        """Create a GameHistoryEntry from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing entry data.
            
        Returns:
            GameHistoryEntry: The reconstructed entry.
        """
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            turn_number=data["turn_number"],
            player=data["player"],
            action_type=ActionType(data["action_type"]),
            card=Card.from_dict(data["card"]) if data["card"] else None,
            target=Card.from_dict(data["target"]) if data["target"] else None,
            source_location=data["source_location"],
            destination_location=data["destination_location"],
            additional_data=data["additional_data"],
            description=data["description"],
        )


class GameHistory:
    """Manages the chronological history of all game actions.
    
    This class provides methods to record game actions and query them for
    testing and analysis purposes. It replaces the need to parse logs by
    providing structured, programmatic access to game events.
    
    Attributes:
        entries (List[GameHistoryEntry]): Chronological list of all game actions.
        turn_counter (int): Current turn number for new entries.
    """
    
    def __init__(self):
        """Initialize an empty game history."""
        self.entries: List[GameHistoryEntry] = []
        self.turn_counter: int = 0
    
    def record_action(
        self,
        player: int,
        action_type: ActionType,
        card: Optional[Card] = None,
        target: Optional[Card] = None,
        source: str = "",
        destination: str = "",
        additional_data: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        """Record a new action in the game history.
        
        Args:
            player (int): The player who performed the action (0 or 1).
            action_type (ActionType): The type of action performed.
            card (Optional[Card]): The primary card involved in the action.
            target (Optional[Card]): The target card (if any) for the action.
            source (str): Where the card came from.
            destination (str): Where the card went.
            additional_data (Optional[Dict[str, Any]]): Additional context data.
            description (str): Human-readable description of the action.
        """
        if additional_data is None:
            additional_data = {}
            
        entry = GameHistoryEntry(
            timestamp=datetime.now(),
            turn_number=self.turn_counter,
            player=player,
            action_type=action_type,
            card=card,
            target=target,
            source_location=source,
            destination_location=destination,
            additional_data=additional_data,
            description=description or self._generate_description(
                player, action_type, card, target, source, destination
            ),
        )
        
        self.entries.append(entry)
    
    def _generate_description(
        self,
        player: int,
        action_type: ActionType,
        card: Optional[Card],
        target: Optional[Card],
        source: str,
        destination: str,
    ) -> str:
        """Generate a human-readable description for an action.
        
        Args:
            player (int): The player who performed the action.
            action_type (ActionType): The type of action.
            card (Optional[Card]): The primary card.
            target (Optional[Card]): The target card.
            source (str): Source location.
            destination (str): Destination location.
            
        Returns:
            str: Human-readable description.
        """
        card_str = str(card) if card else "card"
        target_str = str(target) if target else "target"
        
        if action_type == ActionType.DRAW:
            return f"Player {player} draws {card_str} from {source}"
        elif action_type == ActionType.POINTS:
            points = card.point_value() if card else 0
            return f"Player {player} plays {card_str} for {points} points"
        elif action_type == ActionType.SCUTTLE:
            return f"Player {player} scuttles {target_str} with {card_str}"
        elif action_type == ActionType.ONE_OFF:
            return f"Player {player} plays {card_str} as one-off"
        elif action_type == ActionType.FACE_CARD:
            return f"Player {player} plays {card_str} as face card"
        elif action_type == ActionType.COUNTER:
            return f"Player {player} counters {target_str} with {card_str}"
        elif action_type == ActionType.RESOLVE:
            return f"Player {player} resolves {target_str}"
        elif action_type == ActionType.JACK:
            return f"Player {player} uses {card_str} to steal {target_str}"
        elif action_type == ActionType.TAKE_FROM_DISCARD:
            return f"Player {player} takes {card_str} from discard"
        elif action_type == ActionType.DISCARD_REVEALED:
            return f"Player {player} discards revealed {card_str}"
        elif action_type == ActionType.DISCARD_FROM_HAND:
            return f"Player {player} discards {card_str} from hand"
        else:
            return f"Player {player} performs {action_type.value}"
    
    def increment_turn(self) -> None:
        """Increment the turn counter for new entries."""
        self.turn_counter += 1
    
    def get_actions_by_player(self, player: int) -> List[GameHistoryEntry]:
        """Get all actions performed by a specific player.
        
        Args:
            player (int): The player to filter by (0 or 1).
            
        Returns:
            List[GameHistoryEntry]: List of actions by the specified player.
        """
        return [entry for entry in self.entries if entry.player == player]
    
    def get_actions_by_type(self, action_type: ActionType) -> List[GameHistoryEntry]:
        """Get all actions of a specific type.
        
        Args:
            action_type (ActionType): The action type to filter by.
            
        Returns:
            List[GameHistoryEntry]: List of actions of the specified type.
        """
        return [entry for entry in self.entries if entry.action_type == action_type]
    
    def get_actions_by_turn_range(self, start: int, end: int) -> List[GameHistoryEntry]:
        """Get all actions within a specific turn range.
        
        Args:
            start (int): The starting turn number (inclusive).
            end (int): The ending turn number (inclusive).
            
        Returns:
            List[GameHistoryEntry]: List of actions within the turn range.
        """
        return [
            entry for entry in self.entries
            if start <= entry.turn_number <= end
        ]
    
    def get_last_n_actions(self, n: int) -> List[GameHistoryEntry]:
        """Get the last n actions in chronological order.
        
        Args:
            n (int): The number of recent actions to retrieve.
            
        Returns:
            List[GameHistoryEntry]: List of the last n actions.
        """
        return self.entries[-n:] if n <= len(self.entries) else self.entries[:]
    
    def get_actions_involving_card(self, card: Card) -> List[GameHistoryEntry]:
        """Get all actions involving a specific card (as primary card or target).
        
        Args:
            card (Card): The card to search for.
            
        Returns:
            List[GameHistoryEntry]: List of actions involving the card.
        """
        return [
            entry for entry in self.entries
            if entry.card == card or entry.target == card
        ]
    
    def clear(self) -> None:
        """Clear all history entries and reset turn counter."""
        self.entries.clear()
        self.turn_counter = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the game history to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the game history.
        """
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "turn_counter": self.turn_counter,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameHistory":
        """Create a GameHistory from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing game history data.
            
        Returns:
            GameHistory: The reconstructed game history.
        """
        history = cls()
        history.turn_counter = data["turn_counter"]
        history.entries = [
            GameHistoryEntry.from_dict(entry_data)
            for entry_data in data["entries"]
        ]
        return history
    
    def __len__(self) -> int:
        """Return the number of entries in the history."""
        return len(self.entries)
    
    def __iter__(self):
        """Allow iteration over history entries."""
        return iter(self.entries)
