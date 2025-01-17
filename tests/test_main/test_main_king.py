from unittest.mock import patch
import pytest
from game.card import Card, Suit, Rank
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainKing(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_king_through_main(self, mock_generate_cards, mock_print, mock_input):
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
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards (including Kings and points)
            "0",  # Select King of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select 5 of Diamonds
            "0",  # Select 2 of Clubs
            # Player 1 selects cards
            "0",  # Select 8 of Diamonds
            "0",  # Select 7 of Clubs
            "0",  # Select 6 of Hearts
            "0",  # Select 5 of Spades
            "0",  # Select 4 of Diamonds
            "0",  # Select 3 of Clubs
            "n",  # Don't save initial state
            # Game actions
            "5",  # p0 Play first King (face card)
            "1",  # p1 Play 8 of Diamonds (points)
            "4",  # p0 Play second King (face card)
            "0",  # p1 draws
            "1",  # p0 plays 10 of Hearts (points)
            # Game should be over after this point as Player 0 has won
        ]
        mock_input.side_effect = mock_inputs

        # Import and run main
        from main import main

        main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        # Check for key game events in output
        target_reductions = [
            text
            for text in log_output
            if "Player 0's field: [King of Spades]" in text
            or "Player 1's field: [Eight of Diamonds]" in text
            or "Player 0's field: [King of Spades, King of Hearts]" in text
            or "Player 0's field: [King of Hearts, King of Spades]" in text
            or "Player 0 wins! Score: 10 points (target: 10 with 2 Kings)" in text
        ]
        self.assertTrue(
            any(target_reductions)
        )  # At least one of these messages should appear

        # Check for point accumulation
        point_messages = [text for text in log_output if "10 points" in text]
        self.assertTrue(any(point_messages))

        # Check for win message with points and Kings
        win_messages = [text for text in log_output if "wins!" in text]
        self.assertTrue(len(win_messages) >= 1)  # At least one win message
        final_win = win_messages[-1]  # Get the last win message
        self.assertIn("Player 0", final_win)
        self.assertIn("10 points", final_win)
        self.assertIn("2 Kings", final_win)
