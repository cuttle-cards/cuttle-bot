from unittest.mock import patch
import pytest
from game.card import Card, Suit, Rank
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainQueen(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_queen_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing a Queen through main.py, demonstrating its counter-prevention ability."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.QUEEN),  # Queen of Hearts
            Card("2", Suit.SPADES, Rank.SIX),    # 10 of Spades (points)
            Card("3", Suit.HEARTS, Rank.NINE),   # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE), # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),     # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.TWO),  # 2 of Diamonds (potential counter)
            Card("7", Suit.CLUBS, Rank.SEVEN),    # 7 of Clubs
            Card("8", Suit.HEARTS, Rank.SIX),     # 6 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),    # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR), # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.THREE),   # 3 of Clubs
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

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        self.assertIn("Cannot counter with a two if opponent has a queen on their field", log_output)

        
