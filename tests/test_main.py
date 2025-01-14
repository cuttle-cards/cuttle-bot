import unittest
from unittest.mock import patch, call
import pytest
import sys
import builtins
import logging
import io

from game.card import Card, Suit, Rank


# Set up logging
log_stream = io.StringIO()
logging.basicConfig(
    stream=log_stream, level=logging.DEBUG, format="%(message)s", force=True
)
logger = logging.getLogger(__name__)


def print_and_capture(*args, **kwargs):
    """Helper function to both print to stdout and log the output"""
    # Convert args to string
    output = " ".join(str(arg) for arg in args)
    # Add newline if not present
    if not output.endswith("\n"):
        output += "\n"
    # Write to stdout
    sys.__stdout__.write(output)
    # Log the output (strip to avoid double newlines)
    logger.info(output.rstrip())
    # Return the output for the mock to capture
    return output.rstrip()


class TestMain(unittest.TestCase):
    def setUp(self):
        # Clear the log stream before each test
        log_stream.truncate(0)
        log_stream.seek(0)
        # Reset logging configuration
        logging.basicConfig(
            stream=log_stream, level=logging.DEBUG, format="%(message)s", force=True
        )

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_king_through_main(self, mock_generate_cards, mock_print, mock_input):
        """Test playing a King through main.py using only user inputs."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create a test deck with specific cards in order
        test_deck = [
            # First 5 cards (Player 0's hand)
            Card("1", Suit.HEARTS, Rank.KING),  # King of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
            # Next 6 cards (Player 1's hand)
            Card("6", Suit.DIAMONDS, Rank.EIGHT),  # 8 of Diamonds
            Card("7", Suit.CLUBS, Rank.SEVEN),  # 7 of Clubs
            Card("8", Suit.HEARTS, Rank.SIX),  # 6 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        # Add remaining cards in any order
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                card_str = f"{rank.name} of {suit.name}"
                if not any(str(c) == card_str for c in test_deck):
                    test_deck.append(Card(str(len(test_deck) + 1), suit, rank))

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
        log_output = log_stream.getvalue().splitlines()

        # Print all captured outputs for inspection
        print("\nGame Output:", file=sys.__stdout__, flush=True)
        for i, line in enumerate(log_output):
            print(f"  {i}: {line}", file=sys.__stdout__, flush=True)

        # Print what we're looking for
        print("\nLooking for:", file=sys.__stdout__, flush=True)
        print("  - Player 0's field: [King of Hearts]", file=sys.__stdout__, flush=True)
        print(
            "  - Player 1's field: [Eight of Diamonds]", file=sys.__stdout__, flush=True
        )
        print(
            "  - Player 0's field: [King of Hearts, King of Spades]",
            file=sys.__stdout__,
            flush=True,
        )
        print(
            "  - Player 0 wins! Score: 10 points (target: 10 with 2 Kings)",
            file=sys.__stdout__,
            flush=True,
        )

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

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_six_through_main(self, mock_generate_cards, mock_print, mock_input):
        """Test playing a Six as a one-off through main.py to destroy face cards."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create a test deck with specific cards in order
        test_deck = [
            # First 5 cards (Player 0's hand)
            Card("1", Suit.HEARTS, Rank.SIX),  # Six of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
            # Next 6 cards (Player 1's hand)
            Card("6", Suit.DIAMONDS, Rank.KING),  # King of Diamonds
            Card("7", Suit.CLUBS, Rank.QUEEN),  # Queen of Clubs
            Card("8", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        # Add remaining cards in any order
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                card_str = f"{rank.name} of {suit.name}"
                if not any(str(c) == card_str for c in test_deck):
                    test_deck.append(Card(str(len(test_deck) + 1), suit, rank))

        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
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
            "n",  # Don't save game
            "0",  # p1 resolves
            "e",  # End game
            "n",  # Don't save final game state
        ]
        mock_input.side_effect = mock_inputs

        # Import and run main
        from main import main

        main()

        # Get all logged output
        log_output = log_stream.getvalue().splitlines()

        # Print all captured outputs for inspection
        print("\nGame Output:", file=sys.__stdout__, flush=True)
        for i, line in enumerate(log_output):
            print(f"  {i}: {line}", file=sys.__stdout__, flush=True)

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

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_ace_through_main(self, mock_generate_cards, mock_print, mock_input):
        """Test playing an Ace as a one-off through main.py to destroy point cards."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create a test deck with specific cards in order
        test_deck = [
            # First 5 cards (Player 0's hand)
            Card("1", Suit.HEARTS, Rank.ACE),  # Ace of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
            # Next 6 cards (Player 1's hand)
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds (points)
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs (face)
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts (points)
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        # Add remaining cards in any order
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                card_str = f"{rank.name} of {suit.name}"
                if not any(str(c) == card_str for c in test_deck):
                    test_deck.append(Card(str(len(test_deck) + 1), suit, rank))

        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
            "0",  # Select Ace of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select 5 of Diamonds
            "0",  # Select 2 of Clubs
            # Player 1 selects cards
            "0",  # Select 9 of Diamonds
            "0",  # Select 8 of Clubs
            "0",  # Select 7 of Hearts
            "0",  # Select 5 of Spades
            "0",  # Select 4 of Diamonds
            "0",  # Select 3 of Clubs
            "n",  # Don't save initial state
            # Game actions
            "2",  # p0 Play 10 of Hearts (points)
            "1",  # p1 Play 9 of Diamonds (points)
            "2",  # p0 Play 5 of Diamonds (points)
            "2",  # p1 Play 7 of Hearts (points)
            "4",  # p0 Play Ace of Hearts (one-off) - Counterable
            "n",  # Don't save game
            "0",  # p1 resolves
            "e",  # End game
            "n",  # Don't save final game state
        ]
        mock_input.side_effect = mock_inputs

        # Import and run main
        from main import main

        main()

        # Verify the sequence of prints shows point cards being played and destroyed
        # Get all logged output
        log_output = log_stream.getvalue().splitlines()

        # Print all captured outputs for inspection
        print("\nGame Output:", file=sys.__stdout__, flush=True)
        for i, line in enumerate(log_output):
            print(f"  {i}: {line}", file=sys.__stdout__, flush=True)

        # Check for key game events in output
        point_card_plays = [
            text
            for text in log_output
            if "Player 0's field: [Ten of Hearts]" in text
            or "Player 1's field: [Nine of Diamonds]" in text
            or "Player 0's field: [Ten of Hearts, Five of Diamonds]" in text
            or "Player 1's field: [Nine of Diamonds, Seven of Hearts]" in text
        ]
        self.assertTrue(any(point_card_plays))

        # After Ace is played, fields should be empty of point cards
        empty_fields = [
            text
            for text in log_output
            if "Player 0's field: []" in text or "Player 1's field: []" in text
        ]
        # Get the last occurrence of each empty field
        p0_empty_indices = [
            i for i, text in enumerate(log_output) if "Player 0's field: []" in text
        ]
        p1_empty_indices = [
            i for i, text in enumerate(log_output) if "Player 1's field: []" in text
        ]
        self.assertTrue(
            p0_empty_indices
        )  # Should have at least one empty field state for p0
        self.assertTrue(
            p1_empty_indices
        )  # Should have at least one empty field state for p1
        p0_last_index = p0_empty_indices[-1]
        p1_last_index = p1_empty_indices[-1]
        # The last empty states should be close to each other
        self.assertTrue(
            abs(p0_last_index - p1_last_index) <= 10
        )  # Allow some flexibility in print order

        # Verify final game state
        last_game_state_output = [
            "Deck: 41",
            "Discard Pile: 5",
            "Points:",
            "Player 0: 0",
            "Player 1: 0",
            "Player 0's hand: [King of Spades, Two of Clubs]",
            "Player 1's hand: [Eight of Clubs, Five of Spades, Four of Diamonds, Three of Clubs]",
            "Player 0's field: []",
            "Player 1's field: []",
        ]
        # Check that each line appears in the output
        for expected_line in last_game_state_output:
            self.assertTrue(
                any(expected_line in actual_line for actual_line in log_output[-50:]),
                f"Could not find expected line: {expected_line}",
            )
        # Also verify that these lines appear near the end of the output
        # by checking that all of them appear in the last 50 lines
        last_50_lines = log_output[-50:]
        all_lines_found = all(
            any(expected_line in actual_line for actual_line in last_50_lines)
            for expected_line in last_game_state_output
        )
        self.assertTrue(
            all_lines_found,
            "Not all expected lines were found in the last 50 lines of output",
        )

        self.assertTrue(any(empty_fields))

        # Verify one-off effect message
        ace_effect = [
            text
            for text in log_output
            if "Applying one off effect for Ace of Hearts" in text
        ]
        self.assertTrue(any(ace_effect))


if __name__ == "__main__":
    unittest.main()
