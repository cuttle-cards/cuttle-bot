from unittest.mock import patch
import pytest
from game.card import Card, Suit, Rank, Purpose
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainFour(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_four_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing a Four as a one-off through main.py to force opponent to discard."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.FOUR),  # Four of Hearts
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
            "0",  # Select Four of Hearts
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
            "Four of Hearts as one-off",  # p0 Play Four of Hearts (one-off)
            "0",  # p1 resolves (doesn't counter)
            "0",  # p1 discards first card (9 of Diamonds)
            "0",  # p1 discards second card (8 of Clubs)
            "e",  # End game
            "n",  # Don't save final game state
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        # Verify Four was played
        four_played = [
            text
            for text in log_output
            if "Four of Hearts" in text and "one-off" in text
        ]
        self.assertTrue(any(four_played))

        # Verify opponent had to discard
        discard_prompt = [text for text in log_output if "must discard 2 cards" in text]
        self.assertTrue(any(discard_prompt))

        # Verify cards were discarded
        discarded_cards = [
            text
            for text in log_output
            if "discarded Nine of Diamonds" in text
            or "discarded Eight of Clubs" in text
        ]
        self.assertEqual(len(discarded_cards), 2)

        # Verify final game state
        final_state = [text for text in log_output if "Player 1's hand" in text][-1]
        remaining_cards = [
            "Seven of Hearts",
            "Five of Spades",
            "Four of Diamonds",
            "Three of Clubs",
        ]
        for card in remaining_cards:
            self.assertIn(card, final_state)

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_four_with_counter_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing a Four that gets countered by a Two."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.FOUR),  # Four of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds
            Card("7", Suit.HEARTS, Rank.TWO),  # 2 of Clubs (counter)
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
            "0",  # Select Four of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select 5 of Diamonds
            "0",  # Select 2 of Clubs
            # Player 1 selects cards
            "0",  # Select 9 of Diamonds
            "0",  # Select 2 of Clubs
            "0",  # Select 7 of Hearts
            "0",  # Select 5 of Spades
            "0",  # Select 4 of Diamonds
            "0",  # Select 3 of Clubs
            "n",  # Don't save initial state
            # Game actions
            "Four of Hearts as one-off",  # p0 Play Four of Hearts (one-off)
            "Two of Clubs as counter",  # p1 counters with Two
            "Resolve",  # p0 resolves counter
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

        # Verify Four was played
        four_played = [
            text
            for text in log_output
            if "Four of Hearts" in text and "one-off" in text
        ]
        self.assertTrue(any(four_played))

        # Verify Two was used to counter
        counter_played = [
            text for text in log_output if "Two of Clubs" in text and "Counter" in text
        ]
        self.assertTrue(any(counter_played))

        # Verify no cards were discarded from opponent's hand
        for card in [
            "Nine of Diamonds",
            "Seven of Hearts",
            "Five of Spades",
            "Four of Diamonds",
            "Three of Clubs",
        ]:
            hand_state = [text for text in log_output if "Player 1's hand" in text][-1]
            self.assertIn(card, hand_state)

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_four_with_one_card_opponent_through_main(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing a Four when opponent only has one card to discard."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.FIVE),  # Five of Hearts
            Card("2", Suit.SPADES, Rank.ACE),  # Ace of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.FOUR),  # Four of Diamonds
            Card("7", Suit.CLUBS, Rank.FOUR),  # Four of Clubs
            Card("8", Suit.HEARTS, Rank.FOUR),  # Four of Hearts
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
            "0",  # Select Five of Hearts
            "0",  # Select King of Spades
            "0",  # Select 10 of Hearts
            "0",  # Select 5 of Diamonds
            "0",  # Select 3 of Clubs
            # Player 1 selects cards
            "0",  # Select Four of Diamonds
            "0",  # Select Four of Clubs
            "0",  # Select Four of Hearts
            "done",  # Finish P1's selection
            "n",  # Don't save initial state
            # Game actions
            "Five of Hearts as points",  # p0 Play Five of Hearts as points
            "Four of Diamonds as one-off",  # p1 plays Four of Diamonds one off
            "Resolve",  # p0 resolves
            "0",  # p0 discards first card
            "0",  # p0 discards second card
            "Three of Clubs as points", 
            "Four of Hearts as one-off",  # p1 Play Four of Hearts (one-off)
            "Resolve",  # p0 resolves (doesn't counter)
            "0",  # p0 discards only card
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

        # Verify Four was played
        four_played = [
            text
            for text in log_output
            if "Four of Hearts" in text and "one-off" in text
        ]
        self.assertTrue(any(four_played))

        # Verify opponent had to discard
        discard_prompt = [text for text in log_output if "must discard 1 cards" in text]
        self.assertTrue(any(discard_prompt))

        # Verify card was discarded
        discarded_cards = [
            text for text in log_output if "discarded Five of Diamonds" in text
        ]
        self.assertEqual(len(discarded_cards), 1)

        # Verify final game state - opponent should have no cards
        final_state = [text for text in log_output if "Player 0's hand" in text][-1]
        self.assertIn("[]", final_state)
