import asyncio
import logging
from typing import Any, List
from unittest.mock import Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainAce(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_ace_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
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
            "n",  # Use AI?
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
        
        # Capture the game object using a different approach - monkey patch Game class
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
        
        # Verify point card plays through game history
        points_actions = history.get_actions_by_type(ActionType.POINTS)
        assert len(points_actions) == 4, f"Expected 4 point plays, got {len(points_actions)}"
        
        # Verify the specific cards played for points
        point_cards_played = [action.card for action in points_actions if action.card]
        expected_point_cards = [
            Card("3", Suit.HEARTS, Rank.TEN),   # Ten of Hearts
            Card("6", Suit.DIAMONDS, Rank.NINE), # Nine of Diamonds
            Card("4", Suit.DIAMONDS, Rank.FIVE), # Five of Diamonds  
            Card("8", Suit.HEARTS, Rank.SEVEN),  # Seven of Hearts
        ]
        
        for expected_card in expected_point_cards:
            assert any(card.rank == expected_card.rank and card.suit == expected_card.suit 
                      for card in point_cards_played), f"Expected {expected_card} to be played for points"
        
        # Verify Ace one-off action
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        ace_one_offs = [action for action in one_off_actions 
                       if action.card and action.card.rank == Rank.ACE]
        assert len(ace_one_offs) == 1, "Expected exactly one Ace one-off action"
        ace_action = ace_one_offs[0]
        assert ace_action.card.suit == Suit.HEARTS, "Expected Ace of Hearts to be played"
        assert ace_action.player == 0, "Expected player 0 to play the Ace"
        
        # Verify final game state - both players should have 0 points (Ace destroyed all point cards)
        final_p0_score = sum(card.point_value() for card in captured_game.game_state.fields[0])
        final_p1_score = sum(card.point_value() for card in captured_game.game_state.fields[1])
        assert final_p0_score == 0, f"Player 0 should have 0 points after Ace, got {final_p0_score}"
        assert final_p1_score == 0, f"Player 1 should have 0 points after Ace, got {final_p1_score}"
        
        # Verify fields are empty (all point cards destroyed)
        assert len(captured_game.game_state.fields[0]) == 0, "Player 0's field should be empty"
        assert len(captured_game.game_state.fields[1]) == 0, "Player 1's field should be empty"
        
        # Verify the cards are in discard pile (5 total: 4 point cards + 1 Ace)
        assert len(captured_game.game_state.discard_pile) == 5, f"Expected 5 cards in discard pile, got {len(captured_game.game_state.discard_pile)}"
        
        # Verify hands have the expected remaining cards
        p0_hand = captured_game.game_state.hands[0]
        p1_hand = captured_game.game_state.hands[1]
        assert len(p0_hand) == 2, f"Player 0 should have 2 cards in hand, got {len(p0_hand)}"
        assert len(p1_hand) == 4, f"Player 1 should have 4 cards in hand, got {len(p1_hand)}"
        
        # Verify specific cards in hands
        p0_hand_ranks = {card.rank for card in p0_hand}
        p1_hand_ranks = {card.rank for card in p1_hand}
        assert Rank.KING in p0_hand_ranks and Rank.TWO in p0_hand_ranks, "Player 0 should have King and Two"
        assert Rank.EIGHT in p1_hand_ranks, "Player 1 should have Eight in hand"

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_ace_with_countering_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
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
            "n",  # Use AI?
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
        
        # Capture the game object using a different approach - monkey patch Game class
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

        # Verify point card plays through game history
        points_actions = history.get_actions_by_type(ActionType.POINTS)
        assert len(points_actions) == 4, f"Expected 4 point plays, got {len(points_actions)}"
        
        # Verify Ace one-off action
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        ace_one_offs = [action for action in one_off_actions 
                       if action.card and action.card.rank == Rank.ACE]
        assert len(ace_one_offs) == 1, "Expected exactly one Ace one-off action"
        ace_action = ace_one_offs[0]
        assert ace_action.card.suit == Suit.HEARTS, "Expected Ace of Hearts to be played"
        assert ace_action.player == 0, "Expected player 0 to play the Ace"
        
        # Verify counter action
        counter_actions = history.get_actions_by_type(ActionType.COUNTER)
        assert len(counter_actions) == 1, "Expected exactly one counter action"
        counter_action = counter_actions[0]
        assert counter_action.card.rank == Rank.TWO, "Expected Two to be used for countering"
        assert counter_action.card.suit == Suit.CLUBS, "Expected Two of Clubs to be used"
        assert counter_action.player == 1, "Expected player 1 to counter"
        assert counter_action.target == ace_action.card, "Counter should target the Ace"
        
        # Verify final game state - point cards should still be on the field (Ace was countered)
        final_p0_score = sum(card.point_value() for card in captured_game.game_state.fields[0])
        final_p1_score = sum(card.point_value() for card in captured_game.game_state.fields[1])
        assert final_p0_score == 15, f"Player 0 should have 15 points after countered Ace, got {final_p0_score}"
        assert final_p1_score == 16, f"Player 1 should have 16 points after countered Ace, got {final_p1_score}"
        
        # Verify point cards are still on the field
        p0_field_ranks = {card.rank for card in captured_game.game_state.fields[0]}
        p1_field_ranks = {card.rank for card in captured_game.game_state.fields[1]}
        assert Rank.TEN in p0_field_ranks and Rank.FIVE in p0_field_ranks, "Player 0 should have Ten and Five on field"
        assert Rank.NINE in p1_field_ranks and Rank.SEVEN in p1_field_ranks, "Player 1 should have Nine and Seven on field"
        
        # Verify the Two of Clubs (counter card) is in discard pile
        discard_ranks = {card.rank for card in captured_game.game_state.discard_pile}
        discard_twos = [card for card in captured_game.game_state.discard_pile 
                       if card.rank == Rank.TWO and card.suit == Suit.CLUBS]
        assert len(discard_twos) == 1, "Two of Clubs should be in discard pile"
        
        # Verify Ace is also in discard pile (countered one-offs go to discard)
        discard_aces = [card for card in captured_game.game_state.discard_pile 
                       if card.rank == Rank.ACE and card.suit == Suit.HEARTS]
        assert len(discard_aces) == 1, "Ace of Hearts should be in discard pile"
        
        # Verify discard pile size (2 cards: Ace + Two)
        assert len(captured_game.game_state.discard_pile) == 2, f"Expected 2 cards in discard pile, got {len(captured_game.game_state.discard_pile)}"
        
        # Verify hands have the expected remaining cards
        p0_hand = captured_game.game_state.hands[0]
        p1_hand = captured_game.game_state.hands[1]
        assert len(p0_hand) == 2, f"Player 0 should have 2 cards in hand, got {len(p0_hand)}"
        assert len(p1_hand) == 3, f"Player 1 should have 3 cards in hand, got {len(p1_hand)}"
