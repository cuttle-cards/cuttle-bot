"""
Unit tests for the GameHistory module.

This module contains comprehensive tests for both GameHistoryEntry and GameHistory classes,
including serialization, querying, and integration functionality.
"""

import unittest
from datetime import datetime
from typing import Dict, Any

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game_history import GameHistory, GameHistoryEntry


class TestGameHistoryEntry(unittest.TestCase):
    """Test cases for the GameHistoryEntry class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_card = Card("1", Suit.HEARTS, Rank.ACE)
        self.test_target = Card("2", Suit.SPADES, Rank.KING)
        self.test_timestamp = datetime.now()
        
        self.entry = GameHistoryEntry(
            timestamp=self.test_timestamp,
            turn_number=1,
            player=0,
            action_type=ActionType.POINTS,
            card=self.test_card,
            target=self.test_target,
            source_location="hand",
            destination_location="field",
            additional_data={"test": "value"},
            description="Test action"
        )

    def test_entry_initialization(self) -> None:
        """Test GameHistoryEntry initialization with all fields."""
        self.assertEqual(self.entry.timestamp, self.test_timestamp)
        self.assertEqual(self.entry.turn_number, 1)
        self.assertEqual(self.entry.player, 0)
        self.assertEqual(self.entry.action_type, ActionType.POINTS)
        self.assertEqual(self.entry.card, self.test_card)
        self.assertEqual(self.entry.target, self.test_target)
        self.assertEqual(self.entry.source_location, "hand")
        self.assertEqual(self.entry.destination_location, "field")
        self.assertEqual(self.entry.additional_data, {"test": "value"})
        self.assertEqual(self.entry.description, "Test action")

    def test_entry_minimal_initialization(self) -> None:
        """Test GameHistoryEntry initialization with minimal required fields."""
        minimal_entry = GameHistoryEntry(
            timestamp=self.test_timestamp,
            turn_number=2,
            player=1,
            action_type=ActionType.DRAW
        )
        
        self.assertEqual(minimal_entry.timestamp, self.test_timestamp)
        self.assertEqual(minimal_entry.turn_number, 2)
        self.assertEqual(minimal_entry.player, 1)
        self.assertEqual(minimal_entry.action_type, ActionType.DRAW)
        self.assertIsNone(minimal_entry.card)
        self.assertIsNone(minimal_entry.target)
        self.assertEqual(minimal_entry.source_location, "")
        self.assertEqual(minimal_entry.destination_location, "")
        self.assertEqual(minimal_entry.additional_data, {})
        self.assertEqual(minimal_entry.description, "")

    def test_entry_to_dict(self) -> None:
        """Test conversion of GameHistoryEntry to dictionary."""
        entry_dict = self.entry.to_dict()
        
        expected_keys = {
            "timestamp", "turn_number", "player", "action_type",
            "card", "target", "source_location", "destination_location",
            "additional_data", "description"
        }
        self.assertEqual(set(entry_dict.keys()), expected_keys)
        
        self.assertEqual(entry_dict["timestamp"], self.test_timestamp.isoformat())
        self.assertEqual(entry_dict["turn_number"], 1)
        self.assertEqual(entry_dict["player"], 0)
        self.assertEqual(entry_dict["action_type"], ActionType.POINTS.value)
        self.assertEqual(entry_dict["card"], self.test_card.to_dict())
        self.assertEqual(entry_dict["target"], self.test_target.to_dict())
        self.assertEqual(entry_dict["source_location"], "hand")
        self.assertEqual(entry_dict["destination_location"], "field")
        self.assertEqual(entry_dict["additional_data"], {"test": "value"})
        self.assertEqual(entry_dict["description"], "Test action")

    def test_entry_to_dict_with_none_values(self) -> None:
        """Test conversion of GameHistoryEntry with None values to dictionary."""
        minimal_entry = GameHistoryEntry(
            timestamp=self.test_timestamp,
            turn_number=2,
            player=1,
            action_type=ActionType.DRAW
        )
        
        entry_dict = minimal_entry.to_dict()
        self.assertIsNone(entry_dict["card"])
        self.assertIsNone(entry_dict["target"])

    def test_entry_from_dict(self) -> None:
        """Test creation of GameHistoryEntry from dictionary."""
        entry_dict = self.entry.to_dict()
        restored_entry = GameHistoryEntry.from_dict(entry_dict)
        
        self.assertEqual(restored_entry.timestamp, self.test_timestamp)
        self.assertEqual(restored_entry.turn_number, 1)
        self.assertEqual(restored_entry.player, 0)
        self.assertEqual(restored_entry.action_type, ActionType.POINTS)
        
        # Check card attributes instead of object equality
        self.assertEqual(restored_entry.card.id, self.test_card.id)
        self.assertEqual(restored_entry.card.suit, self.test_card.suit)
        self.assertEqual(restored_entry.card.rank, self.test_card.rank)
        
        # Check target attributes instead of object equality
        self.assertEqual(restored_entry.target.id, self.test_target.id)
        self.assertEqual(restored_entry.target.suit, self.test_target.suit)
        self.assertEqual(restored_entry.target.rank, self.test_target.rank)
        
        self.assertEqual(restored_entry.source_location, "hand")
        self.assertEqual(restored_entry.destination_location, "field")
        self.assertEqual(restored_entry.additional_data, {"test": "value"})
        self.assertEqual(restored_entry.description, "Test action")

    def test_entry_serialization_roundtrip(self) -> None:
        """Test full serialization roundtrip for GameHistoryEntry."""
        entry_dict = self.entry.to_dict()
        restored_entry = GameHistoryEntry.from_dict(entry_dict)
        restored_dict = restored_entry.to_dict()
        
        self.assertEqual(entry_dict, restored_dict)


class TestGameHistory(unittest.TestCase):
    """Test cases for the GameHistory class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.history = GameHistory()
        self.test_card1 = Card("1", Suit.HEARTS, Rank.ACE)
        self.test_card2 = Card("2", Suit.SPADES, Rank.KING)
        self.test_card3 = Card("3", Suit.CLUBS, Rank.QUEEN)

    def test_initial_state(self) -> None:
        """Test GameHistory initial state."""
        self.assertEqual(len(self.history), 0)
        self.assertEqual(self.history.turn_counter, 0)
        self.assertEqual(len(self.history.entries), 0)

    def test_record_action_basic(self) -> None:
        """Test recording a basic action."""
        self.history.record_action(
            player=0,
            action_type=ActionType.DRAW,
            card=self.test_card1,
            source="deck",
            destination="hand"
        )
        
        self.assertEqual(len(self.history), 1)
        entry = self.history.entries[0]
        self.assertEqual(entry.player, 0)
        self.assertEqual(entry.action_type, ActionType.DRAW)
        self.assertEqual(entry.card, self.test_card1)
        self.assertEqual(entry.source_location, "deck")
        self.assertEqual(entry.destination_location, "hand")
        self.assertEqual(entry.turn_number, 0)

    def test_record_action_with_description(self) -> None:
        """Test recording an action with custom description."""
        custom_description = "Custom test action"
        self.history.record_action(
            player=1,
            action_type=ActionType.POINTS,
            card=self.test_card2,
            description=custom_description
        )
        
        self.assertEqual(len(self.history), 1)
        entry = self.history.entries[0]
        self.assertEqual(entry.description, custom_description)

    def test_record_action_auto_description(self) -> None:
        """Test automatic description generation for different action types."""
        test_cases = [
            (ActionType.DRAW, 0, self.test_card1, None, "deck", "hand", "Player 0 draws Ace of Hearts from deck"),
            (ActionType.POINTS, 1, self.test_card2, None, "hand", "field", "Player 1 plays King of Spades for 13 points"),
            (ActionType.SCUTTLE, 0, self.test_card1, self.test_card2, "hand", "discard_pile", "Player 0 scuttles King of Spades with Ace of Hearts"),
            (ActionType.ONE_OFF, 1, self.test_card3, None, "hand", "discard_pile", "Player 1 plays Queen of Clubs as one-off"),
        ]
        
        for action_type, player, card, target, source, dest, expected_desc in test_cases:
            with self.subTest(action_type=action_type):
                history = GameHistory()
                history.record_action(
                    player=player,
                    action_type=action_type,
                    card=card,
                    target=target,
                    source=source,
                    destination=dest
                )
                self.assertEqual(history.entries[0].description, expected_desc)

    def test_increment_turn(self) -> None:
        """Test turn counter increment."""
        self.assertEqual(self.history.turn_counter, 0)
        self.history.increment_turn()
        self.assertEqual(self.history.turn_counter, 1)
        
        # Record action after increment
        self.history.record_action(player=0, action_type=ActionType.DRAW)
        self.assertEqual(self.history.entries[0].turn_number, 1)

    def test_get_actions_by_player(self) -> None:
        """Test filtering actions by player."""
        # Record actions for both players
        self.history.record_action(player=0, action_type=ActionType.DRAW)
        self.history.record_action(player=1, action_type=ActionType.POINTS)
        self.history.record_action(player=0, action_type=ActionType.SCUTTLE)
        self.history.record_action(player=1, action_type=ActionType.ONE_OFF)
        
        player0_actions = self.history.get_actions_by_player(0)
        player1_actions = self.history.get_actions_by_player(1)
        
        self.assertEqual(len(player0_actions), 2)
        self.assertEqual(len(player1_actions), 2)
        
        for action in player0_actions:
            self.assertEqual(action.player, 0)
        for action in player1_actions:
            self.assertEqual(action.player, 1)

    def test_get_actions_by_type(self) -> None:
        """Test filtering actions by type."""
        # Record different types of actions
        self.history.record_action(player=0, action_type=ActionType.DRAW)
        self.history.record_action(player=1, action_type=ActionType.DRAW)
        self.history.record_action(player=0, action_type=ActionType.POINTS)
        self.history.record_action(player=1, action_type=ActionType.SCUTTLE)
        
        draw_actions = self.history.get_actions_by_type(ActionType.DRAW)
        points_actions = self.history.get_actions_by_type(ActionType.POINTS)
        scuttle_actions = self.history.get_actions_by_type(ActionType.SCUTTLE)
        
        self.assertEqual(len(draw_actions), 2)
        self.assertEqual(len(points_actions), 1)
        self.assertEqual(len(scuttle_actions), 1)
        
        for action in draw_actions:
            self.assertEqual(action.action_type, ActionType.DRAW)

    def test_get_actions_by_turn_range(self) -> None:
        """Test filtering actions by turn range."""
        # Record actions across multiple turns
        for turn in range(5):
            self.history.increment_turn()
            self.history.record_action(player=turn % 2, action_type=ActionType.DRAW)
        
        # Test various ranges
        all_actions = self.history.get_actions_by_turn_range(1, 5)
        self.assertEqual(len(all_actions), 5)
        
        first_three = self.history.get_actions_by_turn_range(1, 3)
        self.assertEqual(len(first_three), 3)
        
        single_turn = self.history.get_actions_by_turn_range(3, 3)
        self.assertEqual(len(single_turn), 1)
        self.assertEqual(single_turn[0].turn_number, 3)

    def test_get_last_n_actions(self) -> None:
        """Test getting the last N actions."""
        # Record 5 actions
        for i in range(5):
            self.history.record_action(player=i % 2, action_type=ActionType.DRAW)
        
        # Test getting last 3 actions
        last_three = self.history.get_last_n_actions(3)
        self.assertEqual(len(last_three), 3)
        self.assertEqual(last_three[0], self.history.entries[2])
        self.assertEqual(last_three[1], self.history.entries[3])
        self.assertEqual(last_three[2], self.history.entries[4])
        
        # Test getting more actions than available
        all_actions = self.history.get_last_n_actions(10)
        self.assertEqual(len(all_actions), 5)
        self.assertEqual(all_actions, self.history.entries)

    def test_get_actions_involving_card(self) -> None:
        """Test filtering actions involving a specific card."""
        self.history.record_action(
            player=0, action_type=ActionType.POINTS, card=self.test_card1
        )
        self.history.record_action(
            player=1, action_type=ActionType.SCUTTLE, 
            card=self.test_card2, target=self.test_card1
        )
        self.history.record_action(
            player=0, action_type=ActionType.DRAW, card=self.test_card3
        )
        
        # Actions involving test_card1 (as card or target)
        card1_actions = self.history.get_actions_involving_card(self.test_card1)
        self.assertEqual(len(card1_actions), 2)
        
        # Actions involving test_card3 (only as card)
        card3_actions = self.history.get_actions_involving_card(self.test_card3)
        self.assertEqual(len(card3_actions), 1)

    def test_clear(self) -> None:
        """Test clearing history."""
        # Add some actions
        self.history.record_action(player=0, action_type=ActionType.DRAW)
        self.history.increment_turn()
        self.history.record_action(player=1, action_type=ActionType.POINTS)
        
        # Clear and verify
        self.history.clear()
        self.assertEqual(len(self.history), 0)
        self.assertEqual(self.history.turn_counter, 0)
        self.assertEqual(len(self.history.entries), 0)

    def test_iteration(self) -> None:
        """Test iteration over history entries."""
        actions = [ActionType.DRAW, ActionType.POINTS, ActionType.SCUTTLE]
        for action_type in actions:
            self.history.record_action(player=0, action_type=action_type)
        
        # Test iteration
        iterated_actions = list(self.history)
        self.assertEqual(len(iterated_actions), 3)
        
        for i, entry in enumerate(self.history):
            self.assertEqual(entry.action_type, actions[i])

    def test_history_to_dict(self) -> None:
        """Test conversion of GameHistory to dictionary."""
        self.history.record_action(
            player=0, action_type=ActionType.DRAW, card=self.test_card1
        )
        self.history.increment_turn()
        
        history_dict = self.history.to_dict()
        self.assertIn("entries", history_dict)
        self.assertIn("turn_counter", history_dict)
        self.assertEqual(history_dict["turn_counter"], 1)
        self.assertEqual(len(history_dict["entries"]), 1)

    def test_history_from_dict(self) -> None:
        """Test creation of GameHistory from dictionary."""
        # Create history with actions
        self.history.record_action(
            player=0, action_type=ActionType.DRAW, card=self.test_card1
        )
        self.history.record_action(
            player=1, action_type=ActionType.POINTS, card=self.test_card2
        )
        self.history.increment_turn()
        
        # Serialize and deserialize
        history_dict = self.history.to_dict()
        restored_history = GameHistory.from_dict(history_dict)
        
        self.assertEqual(len(restored_history), 2)
        self.assertEqual(restored_history.turn_counter, 1)
        
        # Check first action
        first_action = restored_history.entries[0]
        self.assertEqual(first_action.player, 0)
        self.assertEqual(first_action.action_type, ActionType.DRAW)
        
        # Check card attributes instead of object equality
        self.assertEqual(first_action.card.id, self.test_card1.id)
        self.assertEqual(first_action.card.suit, self.test_card1.suit)
        self.assertEqual(first_action.card.rank, self.test_card1.rank)

    def test_history_serialization_roundtrip(self) -> None:
        """Test full serialization roundtrip for GameHistory."""
        # Create complex history
        actions_data = [
            (0, ActionType.DRAW, self.test_card1, None),
            (1, ActionType.POINTS, self.test_card2, None),
            (0, ActionType.SCUTTLE, self.test_card3, self.test_card2),
        ]
        
        for player, action_type, card, target in actions_data:
            self.history.record_action(
                player=player, action_type=action_type, card=card, target=target
            )
            self.history.increment_turn()
        
        # Serialize and deserialize
        history_dict = self.history.to_dict()
        restored_history = GameHistory.from_dict(history_dict)
        restored_dict = restored_history.to_dict()
        
        self.assertEqual(history_dict, restored_dict)


class TestGameHistoryDescriptionGeneration(unittest.TestCase):
    """Test cases specifically for automatic description generation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.history = GameHistory()
        self.ace_hearts = Card("1", Suit.HEARTS, Rank.ACE)
        self.king_spades = Card("2", Suit.SPADES, Rank.KING)

    def test_draw_description(self) -> None:
        """Test description generation for DRAW action."""
        self.history.record_action(
            player=0, action_type=ActionType.DRAW, 
            card=self.ace_hearts, source="deck"
        )
        entry = self.history.entries[0]
        self.assertEqual(entry.description, "Player 0 draws Ace of Hearts from deck")

    def test_points_description(self) -> None:
        """Test description generation for POINTS action."""
        self.history.record_action(
            player=1, action_type=ActionType.POINTS, card=self.king_spades
        )
        entry = self.history.entries[0]
        self.assertEqual(entry.description, "Player 1 plays King of Spades for 13 points")

    def test_scuttle_description(self) -> None:
        """Test description generation for SCUTTLE action."""
        self.history.record_action(
            player=0, action_type=ActionType.SCUTTLE,
            card=self.ace_hearts, target=self.king_spades
        )
        entry = self.history.entries[0]
        self.assertEqual(entry.description, "Player 0 scuttles King of Spades with Ace of Hearts")

    def test_counter_description(self) -> None:
        """Test description generation for COUNTER action."""
        self.history.record_action(
            player=1, action_type=ActionType.COUNTER,
            card=self.ace_hearts, target=self.king_spades
        )
        entry = self.history.entries[0]
        self.assertEqual(entry.description, "Player 1 counters King of Spades with Ace of Hearts")

    def test_unknown_action_description(self) -> None:
        """Test description generation for unknown action types."""
        # Use a generic action type
        self.history.record_action(
            player=0, action_type=ActionType.CONCEDE
        )
        entry = self.history.entries[0]
        self.assertEqual(entry.description, "Player 0 performs Concede")


if __name__ == '__main__':
    unittest.main()