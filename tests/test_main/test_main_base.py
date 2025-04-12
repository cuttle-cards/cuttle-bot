import unittest
import sys
import logging
import io
from unittest.mock import patch, MagicMock
import pytest

from game.card import Card, Suit, Rank
from game.game import Game


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


class MainTestBase(unittest.TestCase):
    def setUp(self):
        # Save original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # Redirect stdout/stderr to capture general output if needed
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        # Store mocks passed to tests
        self.mock_input = None
        self.mock_logger = None # Use this for Game logger

    def tearDown(self):
        # Restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.stdout_capture.close()
        self.stderr_capture.close()

    def setup_mock_input(self, mock_input_target, inputs):
        """Helper to set up the mock input sequence."""
        self.mock_input = mock_input_target
        self.mock_input.side_effect = inputs

    def get_captured_stdout(self):
        """Returns captured standard output."""
        return self.stdout_capture.getvalue()

    def get_captured_stderr(self):
        """Returns captured standard error."""
        return self.stderr_capture.getvalue()

    def get_logger_output(self, mock_logger):
        """Helper to get logged output from the mock logger as a single string."""
        if not mock_logger:
            return ""
        # Extract the first argument from each call (assuming simple string logging)
        log_lines = []
        for call in mock_logger.call_args_list:
            args, kwargs = call
            if args:
                log_lines.append(str(args[0]))
            # Could potentially handle kwargs too if needed
        return "\n".join(log_lines)

    def print_game_output(self, output):
        """Helper to print captured output for debugging tests."""
        print("\n--- Game Output ---")
        print(output)
        print("--- End Game Output ---\n")

    def generate_test_deck(self, p0_cards, p1_cards, num_filler=41):
        """Generate a test deck ensuring specific player hands first."""
        deck = list(p0_cards) + list(p1_cards)
        existing_cards = set(str(c) for c in deck)
        
        # Add filler cards, avoiding duplicates
        filler_count = 0
        card_id = len(deck) + 1
        for suit in Suit:
            for rank in Rank:
                if filler_count >= num_filler:
                    break
                card_str = f"{rank.value}{suit.value}" # Use a consistent string representation
                if card_str not in existing_cards:
                    deck.append(Card(str(card_id), suit, rank))
                    existing_cards.add(card_str)
                    filler_count += 1
                    card_id += 1
            if filler_count >= num_filler:
                break
        # Add more unique fillers if needed (e.g., different suits/ranks)
        while filler_count < num_filler:
             # This fallback logic might be needed if standard deck runs out quickly
             # For now, assume 52 cards are enough
             rank = Rank(filler_count % 13 + 1) # Cycle through ranks
             suit = Suit(list(Suit)[filler_count % 4]) # Cycle through suits
             card_str = f"{rank.value}{suit.value}"
             if card_str not in existing_cards:
                  deck.append(Card(str(card_id), suit, rank))
                  existing_cards.add(card_str)
                  filler_count += 1
                  card_id += 1
             else:
                  # If collision, just increment id and try next combo implicitly
                  card_id += 1 
                  # Safety break to prevent infinite loop in edge cases
                  if card_id > 1000: break 

        return deck
