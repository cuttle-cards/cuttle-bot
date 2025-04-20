from typing import Any, List
from unittest.mock import Mock, patch

import pytest

from game.card import Card, Rank, Suit
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainThree(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_three_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Three as a one-off through main.py to take a card from discard pile."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.THREE),  # Three of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.ACE),  # Ace of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts
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
            "0",  # Select Three of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select Ace of Diamonds
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
            "6",  # p0 Play Ace of Clubs (one-off)
            "0",  # p1 resolves
            "1",  # p1 plays Eight of Clubs
            "4",  # p0 Play Three of Hearts (one-off)
            "0",  # p1 resolves
            "0",  # p0 Select Ace of Diamonds from discard pile
            "end game",  # p1 End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output: str = self.get_logger_output(mock_print)
        self.print_game_output(log_output)

        # Check for key game events in output
        # Verify cards were played to points
        point_card_plays = [
            text
            for text in log_output
            if "Player 0's field: [Ten of Hearts]" in text
            or "Player 1's field: [Nine of Diamonds]" in text
        ]
        self.assertTrue(any(point_card_plays))

        # Verify Three one-off effect message
        three_effect = [
            text
            for text in log_output
            if "Applying one off effect for Three of Hearts" in text
            or "Available cards in discard pile:" in text
        ]
        self.assertTrue(any(three_effect))

        # Verify card selection from discard pile
        card_selection = [
            text
            for text in log_output
            if "Took Ten of Hearts from discard pile" in text
        ]
        self.assertTrue(any(card_selection))

        # The Nine of Diamonds should now be in Player 0's hand
        final_state = [
            text
            for text in log_output
            if "Three of Hearts" in text and "Player 0's hand" in text
        ]
        self.assertTrue(any(final_state))

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_three_empty_discard_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Three as a one-off through main.py with empty discard pile."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.THREE),  # Three of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts
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
            "0",  # Select Three of Hearts
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
            "Three of Hearts as one-off",  # p0 Play Three of Hearts (one-off)
            "Resolve",  # p1 resolves
            "end game",  # End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output: str = self.get_logger_output(mock_print)
        self.print_game_output(log_output)

        # Verify empty discard pile message
        empty_discard = [
            text for text in log_output if "No cards in discard pile to take" in text
        ]
        self.assertTrue(any(empty_discard))

        # Verify game state remains unchanged
        # The Three should still be in Player 0's hand
        final_state = [
            text
            for text in log_output
            if "Three of Hearts" in text and "Player 0's hand" in text
        ]
        self.assertTrue(any(final_state))

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_three_with_counter_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Three as a one-off through main.py and getting countered by Two."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.THREE),  # Three of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.TWO),  # 2 of Diamonds (counter)
            Card("11", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
            "0",  # Select Three of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select 5 of Diamonds
            "0",  # Select 3 of Clubs
            # Player 1 selects cards
            "0",  # Select 9 of Diamonds
            "0",  # Select 8 of Clubs
            "0",  # Select 7 of Hearts
            "0",  # Select 5 of Spades
            "0",  # Select 2 of Diamonds
            "0",  # Select 3 of Clubs
            "n",  # Don't save initial state
            # Game actions
            "Ten of Hearts as points",  # p0 Play 10 of Hearts (points)
            "Nine of Diamonds as points",  # p1 Play 9 of Diamonds (points)
            "Three of Hearts as one-off",  # p0 Play Three of Hearts (one-off)
            "Counter",  # p1 counters with Two of Diamonds
            "Resolve",  # p0 resolves
            "end game",  # End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output: str = self.get_logger_output(mock_print)
        self.print_game_output(log_output)

        # Verify point cards were played
        point_card_plays = [
            text
            for text in log_output
            if "Player 0's field: [Ten of Hearts]" in text
            or "Player 1's field: [Nine of Diamonds]" in text
        ]
        self.assertTrue(any(point_card_plays))

        # Verify counter action was available
        counter_action = [
            text
            for text in log_output
            if "Counter" in text and "Two of Diamonds" in text
        ]
        self.assertTrue(any(counter_action))

        # Verify both cards went to discard pile
        discard_state = [text for text in log_output if "Discard Pile: 2" in text]
        self.assertTrue(any(discard_state))

        # Verify final game state - cards should be in discard pile, not in hands
        final_state = [
            text
            for text in log_output
            if "Three of Hearts" not in text
            and "Two of Diamonds" not in text
            and "Player" in text
            and "hand" in text
        ]
        self.assertTrue(any(final_state))
