from unittest.mock import patch
import pytest
from game.card import Card, Suit, Rank
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainSix(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_six_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing a Six as a one-off through main.py to destroy face cards."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.SIX),  # Six of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.KING),  # King of Diamonds
            Card("7", Suit.CLUBS, Rank.QUEEN),  # Queen of Clubs
            Card("8", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
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
            "n",  # Don't use AI player
            # Player 0 selects cards
            "0",  # Select Six of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select 5 of Diamonds
            "0",  # Select 2 of Clubs
            # Player 1 selects cards
            "0",  # Select King of Diamonds
            "0",  # Select Queen of Clubs
            "0",  # Select Jack of Hearts
            "0",  # Select 5 of Spades
            "0",  # Select 4 of Diamonds
            "0",  # Select 3 of Clubs
            "n",  # Don't save initial state
            # Game actions
            "5",  # p0 Play King of Spades (face card)
            "4",  # p1 Play King of Diamonds (face card)
            "5",  # p0 Play Six of Hearts (one-off) - Counterable
            "0",  # p1 resolves
            "end game",  # End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        # Check for key game events in output
        face_card_plays = [
            text
            for text in log_output
            if "Player 0's field: [King of Spades]" in text
            or "Player 1's field: [King of Diamonds]" in text
            or "Player 1's field: [King of Diamonds, Queen of Clubs]" in text
        ]
        self.assertTrue(any(face_card_plays))

        # After Six is played, fields should be empty
        empty_fields = [
            text
            for text in log_output
            if "Player 0's field: []" in text or "Player 1's field: []" in text
        ]
        self.assertTrue(any(empty_fields))
