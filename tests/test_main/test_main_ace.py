from unittest.mock import patch
import pytest
from game.card import Card, Suit, Rank
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainAce(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_ace_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing an Ace as a one-off through main.py to destroy point cards."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.ACE),  # Ace of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds (points)
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs (face)
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts (points)
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
            "Ten of Hearts as points",  # p0 Play 10 of Hearts (points)
            "Nine of Diamonds as points",  # p1 Play 9 of Diamonds (points)
            "Five of Diamonds as points",  # p0 Play 5 of Diamonds (points)
            "Seven of Hearts as points",  # p1 Play 7 of Hearts (points)
            "Ace of Hearts as one-off",  # p0 Play Ace of Hearts (one-off) - Counterable
            "Resolve",  # p1 resolves
            "end game",  # End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Run the game
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

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

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_ace_with_countering_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing an Ace as a one-off through main.py and getting countered."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.ACE),  # Ace of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds (points)
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs (face)
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts (points)
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
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
            "Ten of Hearts as points",  # p0 Play 10 of Hearts (points)
            "Nine of Diamonds as points",  # p1 Play 9 of Diamonds (points)
            "Five of Diamonds as points",  # p0 Play 5 of Diamonds (points)
            "Seven of Hearts as points",  # p1 Play 7 of Hearts (points)
            "Ace of Hearts as one-off",  # p0 Play Ace of Hearts (one-off) - Counterable
            "Counter",  # p1 counters with two of clubs
            "Resolve",  # p0 resolves
            "end game",  # End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Run the game
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

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

        # After Ace is played and countered, fields should be the same point cards
        empty_fields = [
            text
            for text in log_output
            if "Player 0's field: [Ten of Hearts, Five of Diamonds]" in text
            or "Player 1's field: [Nine of Diamonds, Seven of Hearts]" in text
        ]
        # Get the last occurrence of each empty field
        p0_empty_indices = [
            i
            for i, text in enumerate(log_output)
            if "Player 0's field: [Ten of Hearts, Five of Diamonds]" in text
        ]
        p1_empty_indices = [
            i
            for i, text in enumerate(log_output)
            if "Player 1's field: [Nine of Diamonds, Seven of Hearts]" in text
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
            "Discard Pile: 2",
            "Points:",
            "Player 0: 15",
            "Player 1: 16",
            "Player 0's hand: [King of Spades, Three of Clubs]",
            "Player 1's hand: [Eight of Clubs, Five of Spades, Four of Diamonds]",
            "Player 0's field: [Ten of Hearts, Five of Diamonds]",
            "Player 1's field: [Nine of Diamonds, Seven of Hearts]",
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
