"""
Integration tests for GameState history recording functionality.

This module contains tests that verify the integration between GameState
and GameHistory, ensuring that game actions are properly recorded.
"""

import unittest
from typing import List

from game.action import Action, ActionType
from game.card import Card, Rank, Suit
from game.game_history import GameHistory
from game.game_state import GameState


class TestGameStateHistoryIntegration(unittest.TestCase):
    """Test cases for GameState and GameHistory integration."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create test cards
        self.test_cards = [
            Card("1", Suit.HEARTS, Rank.ACE),
            Card("2", Suit.SPADES, Rank.KING),
            Card("3", Suit.CLUBS, Rank.QUEEN),
            Card("4", Suit.DIAMONDS, Rank.TEN),
            Card("5", Suit.HEARTS, Rank.NINE),
        ]
        
        # Set up initial game state
        self.deck = self.test_cards[2:].copy()  # Queen, Ten, Nine
        self.hands = [
            [self.test_cards[0]],  # Player 0: Ace of Hearts
            [self.test_cards[1]],  # Player 1: King of Spades
        ]
        self.fields = [[], []]
        self.discard_pile = []
        
        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

    def test_gamestate_has_history(self) -> None:
        """Test that GameState initializes with GameHistory."""
        self.assertIsInstance(self.game_state.game_history, GameHistory)
        self.assertEqual(len(self.game_state.game_history), 0)
        self.assertEqual(self.game_state.game_history.turn_counter, 0)

    def test_next_turn_increments_history_turn(self) -> None:
        """Test that next_turn increments the history turn counter."""
        initial_turn = self.game_state.game_history.turn_counter
        self.game_state.next_turn()
        self.assertEqual(self.game_state.game_history.turn_counter, initial_turn + 1)

    def test_draw_action_recorded(self) -> None:
        """Test that draw actions are recorded in history."""
        draw_action = Action(ActionType.DRAW, 0)
        self.game_state.update_state(draw_action)
        
        # Check history
        self.assertEqual(len(self.game_state.game_history), 1)
        entry = self.game_state.game_history.entries[0]
        
        self.assertEqual(entry.player, 0)
        self.assertEqual(entry.action_type, ActionType.DRAW)
        self.assertEqual(entry.source_location, "deck")
        self.assertEqual(entry.destination_location, "hand")
        self.assertIsNotNone(entry.timestamp)

    def test_points_action_recorded(self) -> None:
        """Test that points actions are recorded in history."""
        card = self.test_cards[0]  # Ace of Hearts
        points_action = Action(ActionType.POINTS, 0, card=card)
        self.game_state.update_state(points_action)
        
        # Check history
        self.assertEqual(len(self.game_state.game_history), 1)
        entry = self.game_state.game_history.entries[0]
        
        self.assertEqual(entry.player, 0)
        self.assertEqual(entry.action_type, ActionType.POINTS)
        self.assertEqual(entry.card, card)
        self.assertEqual(entry.source_location, "hand")
        self.assertEqual(entry.destination_location, "field")

    def test_scuttle_action_recorded(self) -> None:
        """Test that scuttle actions are recorded in history."""
        # First, play a point card for player 1
        king_card = self.test_cards[1]  # King of Spades  
        self.game_state.turn = 1
        self.game_state.play_points(king_card)
        
        # Now scuttle with player 0
        self.game_state.turn = 0
        ace_card = self.test_cards[0]  # Ace of Hearts
        scuttle_action = Action(ActionType.SCUTTLE, 0, card=ace_card, target=king_card)
        self.game_state.update_state(scuttle_action)
        
        # Check history - should have the scuttle action recorded
        scuttle_entries = self.game_state.game_history.get_actions_by_type(ActionType.SCUTTLE)
        self.assertEqual(len(scuttle_entries), 1)
        
        entry = scuttle_entries[0]
        self.assertEqual(entry.player, 0)
        self.assertEqual(entry.action_type, ActionType.SCUTTLE)
        self.assertEqual(entry.card, ace_card)
        self.assertEqual(entry.target, king_card)
        self.assertEqual(entry.source_location, "hand")
        self.assertEqual(entry.destination_location, "discard_pile")

    def test_multiple_actions_sequence(self) -> None:
        """Test recording a sequence of multiple different actions."""
        actions_sequence = [
            (0, ActionType.DRAW, None, None),
            (1, ActionType.POINTS, self.test_cards[1], None),  # King of Spades
            (0, ActionType.POINTS, self.test_cards[0], None),  # Ace of Hearts
        ]
        
        for player, action_type, card, target in actions_sequence:
            self.game_state.turn = player
            self.game_state.current_action_player = player
            action = Action(action_type, player, card=card, target=target)
            self.game_state.update_state(action)
        
        # Verify all actions were recorded
        self.assertEqual(len(self.game_state.game_history), 3)
        
        # Check each action
        entries = self.game_state.game_history.entries
        
        # First action: Draw
        self.assertEqual(entries[0].action_type, ActionType.DRAW)
        self.assertEqual(entries[0].player, 0)
        
        # Second action: Points (King)
        self.assertEqual(entries[1].action_type, ActionType.POINTS)
        self.assertEqual(entries[1].player, 1)
        self.assertEqual(entries[1].card, self.test_cards[1])
        
        # Third action: Points (Ace)
        self.assertEqual(entries[2].action_type, ActionType.POINTS)
        self.assertEqual(entries[2].player, 0)
        self.assertEqual(entries[2].card, self.test_cards[0])

    def test_history_query_by_player(self) -> None:
        """Test querying history by player after multiple actions."""
        # Record actions for both players
        actions = [
            (0, ActionType.DRAW),
            (1, ActionType.DRAW),
            (0, ActionType.POINTS, self.test_cards[0]),
            (1, ActionType.POINTS, self.test_cards[1]),
        ]
        
        for player, action_type, *args in actions:
            self.game_state.turn = player
            self.game_state.current_action_player = player
            card = args[0] if args else None
            action = Action(action_type, player, card=card)
            self.game_state.update_state(action)
        
        # Query by player
        player0_actions = self.game_state.game_history.get_actions_by_player(0)
        player1_actions = self.game_state.game_history.get_actions_by_player(1)
        
        self.assertEqual(len(player0_actions), 2)
        self.assertEqual(len(player1_actions), 2)
        
        # Verify player 0 actions
        self.assertEqual(player0_actions[0].action_type, ActionType.DRAW)
        self.assertEqual(player0_actions[1].action_type, ActionType.POINTS)

    def test_history_query_by_action_type(self) -> None:
        """Test querying history by action type after multiple actions."""
        # Record different types of actions
        self.game_state.update_state(Action(ActionType.DRAW, 0))
        self.game_state.turn = 1
        self.game_state.update_state(Action(ActionType.DRAW, 1))
        
        # Query by action type
        draw_actions = self.game_state.game_history.get_actions_by_type(ActionType.DRAW)
        points_actions = self.game_state.game_history.get_actions_by_type(ActionType.POINTS)
        
        self.assertEqual(len(draw_actions), 2)
        self.assertEqual(len(points_actions), 0)

    def test_multi_card_draw_recording(self) -> None:
        """Test that multi-card draws are properly recorded."""
        # Test drawing multiple cards (like from a 5 card effect)
        initial_deck_size = len(self.game_state.deck)
        self.game_state.draw_card(count=2)
        
        # Check that deck has 2 fewer cards
        self.assertEqual(len(self.game_state.deck), initial_deck_size - 2)
        
        # Check that individual draws were recorded for multi-card draws
        draw_actions = self.game_state.game_history.get_actions_by_type(ActionType.DRAW)
        self.assertEqual(len(draw_actions), 2)  # Two individual card draws recorded


class TestGameStateHistorySerialization(unittest.TestCase):
    """Test cases for GameState serialization with history."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_card = Card("1", Suit.HEARTS, Rank.ACE)
        self.deck = [Card("2", Suit.SPADES, Rank.KING)]
        self.hands = [[self.test_card], []]
        self.fields = [[], []]
        self.discard_pile = []
        
        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

    def test_gamestate_serialization_includes_history(self) -> None:
        """Test that GameState.to_dict() includes game history."""
        # Record some actions
        self.game_state.update_state(Action(ActionType.DRAW, 0))
        self.game_state.update_state(Action(ActionType.POINTS, 0, card=self.test_card))
        
        # Serialize
        state_dict = self.game_state.to_dict()
        
        # Check that history is included
        self.assertIn("game_history", state_dict)
        self.assertIsInstance(state_dict["game_history"], dict)
        
        # Check history content
        history_dict = state_dict["game_history"]
        self.assertIn("entries", history_dict)
        self.assertIn("turn_counter", history_dict)
        self.assertEqual(len(history_dict["entries"]), 2)

    def test_gamestate_deserialization_restores_history(self) -> None:
        """Test that GameState.from_dict() properly restores game history."""
        # Record some actions
        original_actions = [
            Action(ActionType.DRAW, 0),
            Action(ActionType.POINTS, 0, card=self.test_card),
        ]
        
        for action in original_actions:
            self.game_state.update_state(action)
        
        # Serialize and deserialize
        state_dict = self.game_state.to_dict()
        restored_state = GameState.from_dict(state_dict)
        
        # Check that history was restored
        self.assertIsInstance(restored_state.game_history, GameHistory)
        self.assertEqual(len(restored_state.game_history), 2)
        
        # Check that actions are the same
        original_entries = self.game_state.game_history.entries
        restored_entries = restored_state.game_history.entries
        
        for orig, restored in zip(original_entries, restored_entries):
            self.assertEqual(orig.player, restored.player)
            self.assertEqual(orig.action_type, restored.action_type)
            # Compare card attributes if both cards exist
            if orig.card and restored.card:
                self.assertEqual(orig.card.id, restored.card.id)
                self.assertEqual(orig.card.suit, restored.card.suit)
                self.assertEqual(orig.card.rank, restored.card.rank)
            else:
                self.assertEqual(orig.card, restored.card)  # Both should be None

    def test_gamestate_deserialization_without_history(self) -> None:
        """Test GameState.from_dict() with data that doesn't include history."""
        # Create state dict without history (simulating old save files)
        state_dict = {
            "hands": [[self.test_card.to_dict()], []],
            "fields": [[], []],
            "deck": [card.to_dict() for card in self.deck],
            "discard_pile": [],
            "turn": 0,
            "current_action_player": 0,
            "status": None,
            "resolving_two": False,
            "resolving_one_off": False,
            "resolving_three": False,
            "one_off_card_to_counter": None,
            "use_ai": False,
            "overall_turn": 0,
            # Note: no "game_history" key
        }
        
        # Should create with empty history
        restored_state = GameState.from_dict(state_dict)
        self.assertIsInstance(restored_state.game_history, GameHistory)
        self.assertEqual(len(restored_state.game_history), 0)

    def test_gamestate_serialization_roundtrip_with_history(self) -> None:
        """Test full serialization roundtrip preserves history."""
        # Add more cards to deck for multiple draws
        extra_cards = [
            Card("3", Suit.CLUBS, Rank.QUEEN),
            Card("4", Suit.DIAMONDS, Rank.TEN),
        ]
        self.game_state.deck.extend(extra_cards)
        
        # Record a complex sequence of actions
        actions = [
            Action(ActionType.DRAW, 0),
            Action(ActionType.POINTS, 0, card=self.test_card),
        ]
        
        for action in actions:
            if action.action_type == ActionType.POINTS:
                # For points, ensure card is in hand first
                if self.test_card not in self.game_state.hands[0]:
                    self.game_state.hands[0].append(self.test_card)
            
            self.game_state.turn = action.played_by
            self.game_state.current_action_player = action.played_by
            self.game_state.update_state(action)
        
        # Serialize and deserialize
        state_dict = self.game_state.to_dict()
        restored_state = GameState.from_dict(state_dict)
        
        # Serialize again
        restored_dict = restored_state.to_dict()
        
        # History sections should be identical
        self.assertEqual(state_dict["game_history"], restored_dict["game_history"])


class TestGameStateHistoryEdgeCases(unittest.TestCase):
    """Test edge cases for GameState history integration."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Add some cards to deck to prevent empty deck issues
        test_deck = [
            Card("1", Suit.HEARTS, Rank.ACE),
            Card("2", Suit.SPADES, Rank.KING),
        ]
        self.game_state = GameState([[]], [[]], test_deck, [])

    def test_history_with_none_card(self) -> None:
        """Test history recording with None card values."""
        action = Action(ActionType.DRAW, 0, card=None)
        self.game_state.update_state(action)
        
        self.assertEqual(len(self.game_state.game_history), 1)
        entry = self.game_state.game_history.entries[0]
        self.assertIsNone(entry.card)

    def test_history_with_none_target(self) -> None:
        """Test history recording with None target values."""
        test_card = Card("1", Suit.HEARTS, Rank.ACE)
        action = Action(ActionType.POINTS, 0, card=test_card, target=None)
        self.game_state.hands[0].append(test_card)
        self.game_state.update_state(action)
        
        self.assertEqual(len(self.game_state.game_history), 1)
        entry = self.game_state.game_history.entries[0]
        self.assertIsNone(entry.target)

    def test_history_turn_sync_edge_cases(self) -> None:
        """Test turn synchronization in edge cases."""
        # Test multiple turn advances
        for _ in range(10):
            self.game_state.next_turn()
        
        # History turn counter should match
        expected_turn = 10
        self.assertEqual(self.game_state.game_history.turn_counter, expected_turn)

    def test_empty_history_queries(self) -> None:
        """Test querying empty history."""
        # All query methods should return empty results
        self.assertEqual(len(self.game_state.game_history.get_actions_by_player(0)), 0)
        self.assertEqual(len(self.game_state.game_history.get_actions_by_type(ActionType.DRAW)), 0)
        self.assertEqual(len(self.game_state.game_history.get_actions_by_turn_range(0, 10)), 0)
        self.assertEqual(len(self.game_state.game_history.get_last_n_actions(5)), 0)


if __name__ == '__main__':
    unittest.main()