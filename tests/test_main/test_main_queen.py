from typing import Any, List
from unittest.mock import Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainQueen(MainTestBase):
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_queen_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Queen through main.py, demonstrating its counter-prevention ability."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.QUEEN),  # Queen of Hearts
            Card("2", Suit.SPADES, Rank.SIX),  # 10 of Spades (points)
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.TWO),  # 2 of Diamonds (potential counter)
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
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
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
            "Play Queen of Hearts as face card",  # p0 Play Queen of Hearts (face card)
            "Play Seven of Clubs as points",  # p1 plays seven as points
            "Play Six of Spades as one-off",  # p0 plays 6 of Spades as one-off
            "Resolve one-off Six of Spades",  # resolve
            "end game",  # end game
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
            await main()
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Queen was played as face card
        face_card_actions = history.get_actions_by_type(ActionType.FACE_CARD)
        queen_actions = [action for action in face_card_actions 
                        if action.card and action.card.rank == Rank.QUEEN]
        assert len(queen_actions) == 1, "Expected exactly one Queen face card action"
        queen_action = queen_actions[0]
        assert queen_action.card.suit == Suit.HEARTS, "Expected Queen of Hearts to be played"
        assert queen_action.player == 0, "Expected player 0 to play the Queen"
        
        # Verify Six was played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        six_one_offs = [action for action in one_off_actions 
                       if action.card and action.card.rank == Rank.SIX]
        assert len(six_one_offs) == 1, "Expected exactly one Six one-off action"
        six_action = six_one_offs[0]
        assert six_action.card.suit == Suit.SPADES, "Expected Six of Spades to be played"
        assert six_action.player == 0, "Expected player 0 to play the Six"
        
        # Verify no counter actions occurred (Queen prevents counters)
        counter_actions = history.get_actions_by_type(ActionType.COUNTER)
        assert len(counter_actions) == 0, "No counter actions should occur when Queen is on field"
        
        # Verify final game state - Player 0 should have Queen on field
        p0_field = captured_game.game_state.fields[0]
        queens_on_field = [card for card in p0_field if card.rank == Rank.QUEEN]
        assert len(queens_on_field) == 1, "Player 0 should have Queen on field"
        
        # Verify Player 1 still has Two in hand (couldn't use it to counter)
        p1_hand = captured_game.game_state.hands[1]
        twos_in_hand = [card for card in p1_hand if card.rank == Rank.TWO]
        assert len(twos_in_hand) >= 1, "Player 1 should still have Two in hand (couldn't counter)"
