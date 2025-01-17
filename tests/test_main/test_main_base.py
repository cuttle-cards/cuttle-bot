import unittest
from unittest.mock import patch
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


class MainTestBase(unittest.TestCase):
    def setUp(self):
        # Clear the log stream before each test
        log_stream.truncate(0)
        log_stream.seek(0)
        # Reset logging configuration
        logging.basicConfig(
            stream=log_stream, level=logging.DEBUG, format="%(message)s", force=True
        )

    def generate_test_deck(self, p0_cards, p1_cards):
        """Helper method to generate a test deck with specified cards for each player."""
        test_deck = p0_cards + p1_cards
        # Add remaining cards in any order
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                card_str = f"{rank.name} of {suit.name}"
                if not any(str(c) == card_str for c in test_deck):
                    test_deck.append(Card(str(len(test_deck) + 1), suit, rank))
        return test_deck

    def get_log_output(self):
        """Helper method to get all logged output as a list of lines."""
        return log_stream.getvalue().splitlines()

    def print_game_output(self, log_output):
        """Helper method to print game output for debugging."""
        print("\nGame Output:", file=sys.__stdout__, flush=True)
        for i, line in enumerate(log_output):
            print(f"  {i}: {line}", file=sys.__stdout__, flush=True)
