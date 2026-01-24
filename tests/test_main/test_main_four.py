import asyncio
from unittest.mock import Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainFour(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_four_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
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
            "n",  # Don't use AI
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
            "0",  # p0 draws a card or passes
            "Play Four of Diamonds as one-off",  # p1 Play Four of Diamonds as one-off
            "0",  # p0 resolves (doesn't counter)
            "0",  # p0 discards first card
            "0",  # p0 discards second card
            "e",  # End game
            "n",  # Don't save final game state
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
            asyncio.run(main())
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Four was played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        four_one_offs = [action for action in one_off_actions 
                        if action.card and action.card.rank == Rank.FOUR]
        assert len(four_one_offs) == 1, "Expected exactly one Four one-off action"
        four_action = four_one_offs[0]
        assert four_action.card.suit == Suit.DIAMONDS, "Expected Four of Diamonds to be played"
        assert four_action.player == 1, "Expected player 1 to play the Four"
        
        # Verify final game state - Player 1 should have 5 remaining cards (Four of Diamonds was played)
        p1_hand = captured_game.game_state.hands[1]
        assert len(p1_hand) == 5, f"Player 1 should have 5 cards remaining, got {len(p1_hand)}"
        
        # Verify specific cards are still in Player 1's hand
        p1_hand_ranks = {card.rank for card in p1_hand}
        expected_remaining = {Rank.NINE, Rank.EIGHT, Rank.SEVEN, Rank.FIVE, Rank.THREE}
        assert p1_hand_ranks == expected_remaining, f"Expected Player 1 to have {expected_remaining}, got {p1_hand_ranks}"

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_four_with_counter_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
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
            "n",  # Don't use AI
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
            "Play Four of Hearts as one-off",  # p0 Play Four of Hearts (one-off)
            "Counter Four of Hearts with Two of Hearts",  # p1 counters with Two of Hearts
            "Resolve",  # p0 resolves counter
            "end game",  # End game
            "n",  # Don't save final game state
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
            asyncio.run(main())
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Four was played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        four_one_offs = [action for action in one_off_actions 
                        if action.card and action.card.rank == Rank.FOUR]
        assert len(four_one_offs) == 1, "Expected exactly one Four one-off action"
        four_action = four_one_offs[0]
        assert four_action.card.suit == Suit.HEARTS, "Expected Four of Hearts to be played"
        assert four_action.player == 0, "Expected player 0 to play the Four"
        
        # Verify counter action
        counter_actions = history.get_actions_by_type(ActionType.COUNTER)
        assert len(counter_actions) == 1, "Expected exactly one counter action"
        counter_action = counter_actions[0]
        assert counter_action.card.rank == Rank.TWO, "Expected Two to be used for countering"
        assert counter_action.card.suit == Suit.HEARTS, "Expected Two of Hearts to be used"
        assert counter_action.player == 1, "Expected player 1 to counter"
        assert counter_action.target == four_action.card, "Counter should target the Four"
        
        # Verify no cards were discarded from opponent's hand (counter prevented effect)
        p1_hand = captured_game.game_state.hands[1]
        assert len(p1_hand) == 5, f"Player 1 should have 5 cards remaining (no discards due to counter), got {len(p1_hand)}"
        
        # Verify specific cards are still in Player 1's hand
        p1_hand_ranks = {card.rank for card in p1_hand}
        expected_remaining = {Rank.NINE, Rank.SEVEN, Rank.FIVE, Rank.FOUR, Rank.THREE}
        assert p1_hand_ranks == expected_remaining, f"Expected Player 1 to have {expected_remaining}, got {p1_hand_ranks}"

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_four_with_one_card_opponent_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
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
            "n",  # Don't use AI
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
            asyncio.run(main())
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Four actions were played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        four_one_offs = [action for action in one_off_actions 
                        if action.card and action.card.rank == Rank.FOUR]
        assert len(four_one_offs) >= 1, "Expected at least one Four one-off action"
        
        # Find the Four of Hearts action by Player 1
        p1_four_hearts = [action for action in four_one_offs 
                         if action.player == 1 and action.card.suit == Suit.HEARTS]
        assert len(p1_four_hearts) == 1, "Expected Player 1 to play Four of Hearts"
        
        # Verify final game state - Player 0 should have minimal cards remaining
        p0_hand = captured_game.game_state.hands[0]
        assert len(p0_hand) <= 1, f"Player 0 should have 1 or fewer cards remaining, got {len(p0_hand)}"

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_four_with_empty_opponent_hand_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Four as a one-off when opponent has no cards in hand."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.FOUR),  # Four of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("5", Suit.CLUBS, Rank.FOUR),  # 4 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.NINE),  # 9 of Diamonds
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs
            Card("8", Suit.HEARTS, Rank.SEVEN),  # 7 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("11", Suit.CLUBS, Rank.THREE),  # 3 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't use AI
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
            # First, make Player 1 play all their cards as points to empty their hand
            "Four of Diamonds as one-off",  # p0 plays 4 of Diamonds as points
            "Resolve",  # p1 resolves
            "0",  # p1 discards first card
            "0",  # p1 discards second card
            "Seven of Hearts as points",  # p1 plays 7 of Hearts as points
            "Four of Hearts as one-off",  # p0 plays 4 of Hearts as one-off
            "Resolve",  # p1 resolves
            "0",  # p1 discards first card
            "0",  # p1 discards second card
            "Three of Clubs as points",  # p1 plays 3 of Clubs as points
            "Four of Clubs as one-off",  # p0 plays 4 of Clubs as one-off
            "Resolve",  # p1 resolves
            # p1 has no cards to discard
            "end game",  # End game
            "n",  # Don't save final game state
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
            asyncio.run(main())
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify multiple Four cards were played as one-offs
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        four_one_offs = [action for action in one_off_actions 
                        if action.card and action.card.rank == Rank.FOUR]
        assert len(four_one_offs) >= 2, f"Expected at least 2 Four one-off actions, got {len(four_one_offs)}"
        
        # Verify at least one Four was played by Player 0
        p0_four_actions = [action for action in four_one_offs if action.player == 0]
        assert len(p0_four_actions) >= 1, "Expected Player 0 to play at least one Four"
        
        # Verify final game state - Player 1 should have minimal cards remaining
        p1_hand = captured_game.game_state.hands[1]
        assert len(p1_hand) <= 1, f"Player 1 should have very few cards remaining, got {len(p1_hand)}"
