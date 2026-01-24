import asyncio
from unittest.mock import Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainThree(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_three_through_main(
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
            "n",  # Don't use AI
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
            "Ten of Hearts as points",  # p0 Play 10 of Hearts (points)
            "Nine of Diamonds as points",  # p1 Play 9 of Diamonds (points)
            "Ace of Diamonds as one-off",  # p0 Play Ace of Diamonds (one-off)
            "0",  # p1 resolves
            "Eight of Clubs as points",  # p1 plays Eight of Clubs
            "Three of Hearts as one-off",  # p0 Play Three of Hearts (one-off)
            "0",  # p1 resolves
            "Ace of Diamonds",  # p0 Select Ace of Diamonds from discard pile
            "end game",  # p1 End game
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
        
        # Verify points were played
        points_actions = history.get_actions_by_type(ActionType.POINTS)
        ten_points = [action for action in points_actions 
                     if action.card and action.card.rank == Rank.TEN]
        nine_points = [action for action in points_actions 
                      if action.card and action.card.rank == Rank.NINE]
        assert len(ten_points) == 1, "Expected Ten of Hearts to be played for points"
        assert len(nine_points) == 1, "Expected Nine of Diamonds to be played for points"
        assert ten_points[0].player == 0, "Expected player 0 to play Ten"
        assert nine_points[0].player == 1, "Expected player 1 to play Nine"
        
        # Verify Ace was played as one-off (destroying all point cards)
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        ace_one_offs = [action for action in one_off_actions 
                       if action.card and action.card.rank == Rank.ACE]
        assert len(ace_one_offs) == 1, "Expected exactly one Ace one-off action"
        assert ace_one_offs[0].player == 0, "Expected player 0 to play Ace"
        
        # Verify Three was played as one-off 
        three_one_offs = [action for action in one_off_actions 
                         if action.card and action.card.rank == Rank.THREE]
        assert len(three_one_offs) == 1, "Expected exactly one Three one-off action"
        three_action = three_one_offs[0]
        assert three_action.card.suit == Suit.HEARTS, "Expected Three of Hearts to be played"
        assert three_action.player == 0, "Expected player 0 to play Three"
        
        # Verify final game state - Player 0 should have retrieved card from discard
        p0_hand = captured_game.game_state.hands[0]
        ace_in_hand = [card for card in p0_hand if card.rank == Rank.ACE]
        assert len(ace_in_hand) == 1, "Player 0 should have retrieved Ace from discard pile"

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_three_empty_discard_through_main(
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
            "n",  # Don't use AI
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
        
        # Verify Three was played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        three_one_offs = [action for action in one_off_actions 
                         if action.card and action.card.rank == Rank.THREE]
        assert len(three_one_offs) == 1, "Expected exactly one Three one-off action"
        three_action = three_one_offs[0]
        assert three_action.card.suit == Suit.HEARTS, "Expected Three of Hearts to be played"
        assert three_action.player == 0, "Expected player 0 to play Three"
        
        # Verify discard pile is empty (no cards to retrieve)
        discard_pile = captured_game.game_state.discard_pile
        assert len(discard_pile) == 1, "Only the Three should be in discard pile after playing"
        
        # Verify Player 0's hand doesn't have any new cards (Three effect failed)
        p0_hand = captured_game.game_state.hands[0]
        assert len(p0_hand) == 4, "Player 0 should have 4 cards (originally 5, played 1)"
        
        # Verify Three is in discard pile
        three_in_discard = [card for card in discard_pile if card.rank == Rank.THREE and card.suit == Suit.HEARTS]
        assert len(three_in_discard) == 1, "Three of Hearts should be in discard pile"

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_three_with_counter_through_main(
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
            "n",  # Don't use AI
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
        
        # Verify points were played
        points_actions = history.get_actions_by_type(ActionType.POINTS)
        ten_points = [action for action in points_actions 
                     if action.card and action.card.rank == Rank.TEN]
        nine_points = [action for action in points_actions 
                      if action.card and action.card.rank == Rank.NINE]
        assert len(ten_points) == 1, "Expected Ten of Hearts to be played for points"
        assert len(nine_points) == 1, "Expected Nine of Diamonds to be played for points"
        
        # Verify Three was played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        three_one_offs = [action for action in one_off_actions 
                         if action.card and action.card.rank == Rank.THREE]
        assert len(three_one_offs) == 1, "Expected exactly one Three one-off action"
        three_action = three_one_offs[0]
        assert three_action.card.suit == Suit.HEARTS, "Expected Three of Hearts to be played"
        assert three_action.player == 0, "Expected player 0 to play Three"
        
        # Verify counter action
        counter_actions = history.get_actions_by_type(ActionType.COUNTER)
        assert len(counter_actions) == 1, "Expected exactly one counter action"
        counter_action = counter_actions[0]
        assert counter_action.card.rank == Rank.TWO, "Expected Two to be used for countering"
        assert counter_action.card.suit == Suit.DIAMONDS, "Expected Two of Diamonds to be used"
        assert counter_action.player == 1, "Expected player 1 to counter"
        assert counter_action.target == three_action.card, "Counter should target the Three"
        
        # Verify both cards are in discard pile (countered one-offs go to discard)
        discard_pile = captured_game.game_state.discard_pile
        three_in_discard = [card for card in discard_pile if card.rank == Rank.THREE and card.suit == Suit.HEARTS]
        two_in_discard = [card for card in discard_pile if card.rank == Rank.TWO and card.suit == Suit.DIAMONDS]
        assert len(three_in_discard) == 1, "Three of Hearts should be in discard pile"
        assert len(two_in_discard) == 1, "Two of Diamonds should be in discard pile"
        
        # Verify point cards are still on the field (Three was countered)
        p0_field = captured_game.game_state.fields[0]
        p1_field = captured_game.game_state.fields[1]
        ten_on_field = [card for card in p0_field if card.rank == Rank.TEN]
        nine_on_field = [card for card in p1_field if card.rank == Rank.NINE]
        assert len(ten_on_field) == 1, "Ten of Hearts should still be on Player 0's field"
        assert len(nine_on_field) == 1, "Nine of Diamonds should still be on Player 1's field"
