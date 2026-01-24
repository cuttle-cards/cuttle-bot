import asyncio
from unittest.mock import Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainKing(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_king_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a King through main.py using only user inputs."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.KING),  # King of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.EIGHT),  # 8 of Diamonds
            Card("7", Suit.CLUBS, Rank.SEVEN),  # 7 of Clubs
            Card("8", Suit.HEARTS, Rank.SIX),  # 6 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't use AI
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards (including Kings and points)
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 0
            # Player 1 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 1
            "n",  # Don't save initial state
            # Game actions
            "King of Hearts as face card",  # p0 Play first King (face card)
            "Eight of Diamonds as points",  # p1 Play 8 of Diamonds (points)
            "King of Spades as face card",  # p0 Play second King (face card)
            "Draw",  # p1 draws
            "Ten of Hearts as points",  # p0 plays 10 of Hearts (points)
            "n",  # Don't save game history
        ]
        self.setup_mock_input(mock_input, mock_inputs)
        
        # Capture the game object using monkey patching
        captured_game = None
        original_init = Game.__init__
        
        def capture_game_init(self, *args, **kwargs):
            nonlocal captured_game
            result = original_init(self, *args, **kwargs)
            captured_game = self
            return result
        
        # Monkey patch temporarily
        Game.__init__ = capture_game_init
        
        try:
            # Run the game
            from main import main
            asyncio.run(main())
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Kings were played as face cards
        face_card_actions = history.get_actions_by_type(ActionType.FACE_CARD)
        king_actions = [action for action in face_card_actions 
                       if action.card and action.card.rank == Rank.KING]
        assert len(king_actions) == 2, f"Expected 2 King face card actions, got {len(king_actions)}"
        
        # Verify both Kings were played by Player 0
        for king_action in king_actions:
            assert king_action.player == 0, "Expected player 0 to play Kings"
            assert king_action.card.suit in [Suit.HEARTS, Suit.SPADES], "Expected King of Hearts or Spades"
        
        # Verify points were played
        points_actions = history.get_actions_by_type(ActionType.POINTS)
        ten_points = [action for action in points_actions 
                     if action.card and action.card.rank == Rank.TEN]
        assert len(ten_points) == 1, "Expected Ten of Hearts to be played for points"
        assert ten_points[0].player == 0, "Expected player 0 to play Ten of Hearts"
        
        # Verify final game state - Player 0 should have 2 Kings on field reducing target
        p0_field = captured_game.game_state.fields[0]
        kings_on_field = [card for card in p0_field if card.rank == Rank.KING]
        assert len(kings_on_field) == 2, f"Player 0 should have 2 Kings on field, got {len(kings_on_field)}"
        
        # Verify Player 0 has enough points to win with reduced target
        p0_score = sum(card.point_value() for card in p0_field if card.rank != Rank.KING)
        effective_target = captured_game.game_state.get_player_target(0)
        assert effective_target == 10, f"Player 0 should have target 10, got {effective_target}"
        assert p0_score >= effective_target, f"Player 0 should have won with score {p0_score} vs target {effective_target}"
