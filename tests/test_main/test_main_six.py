import asyncio
from unittest.mock import Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainSix(MainTestBase):
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    def test_play_six_through_main(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Six as a one-off through main.py to destroy face cards."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.SIX),  # Six of Hearts
            Card("2", Suit.SPADES, Rank.KING),  # King of Spades
            Card("3", Suit.HEARTS, Rank.TEN),  # 10 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.KING),  # King of Diamonds
            Card("7", Suit.CLUBS, Rank.QUEEN),  # Queen of Clubs
            Card("8", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
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
            "King of Spades as face card",  # p0 Play King of Spades (face card)
            "King of Diamonds as face card",  # p1 Play King of Diamonds (face card)
            "Six of Hearts as one-off",  # p0 Play Six of Hearts (one-off) - Counterable
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
        
        # Verify face cards were played
        face_card_actions = history.get_actions_by_type(ActionType.FACE_CARD)
        king_actions = [action for action in face_card_actions 
                       if action.card and action.card.rank == Rank.KING]
        queen_actions = [action for action in face_card_actions 
                        if action.card and action.card.rank == Rank.QUEEN]
        assert len(king_actions) == 2, "Expected exactly two King face card actions"
        # Verify Six was played as one-off
        one_off_actions = history.get_actions_by_type(ActionType.ONE_OFF)
        six_one_offs = [action for action in one_off_actions 
                       if action.card and action.card.rank == Rank.SIX]
        assert len(six_one_offs) == 1, "Expected exactly one Six one-off action"
        six_action = six_one_offs[0]
        assert six_action.card.suit == Suit.HEARTS, "Expected Six of Hearts to be played"
        assert six_action.player == 0, "Expected player 0 to play Six"
        
        # Verify final game state - face cards should be destroyed by Six
        p0_field = captured_game.game_state.fields[0]
        p1_field = captured_game.game_state.fields[1]
        total_face_cards = 0
        for card in p0_field + p1_field:
            if card.rank in [Rank.KING, Rank.QUEEN, Rank.JACK]:
                total_face_cards += 1
        assert total_face_cards == 0, "All face cards should be destroyed by Six"
        
        # Verify face cards are in discard pile
        discard_pile = captured_game.game_state.discard_pile
        face_cards_in_discard = [card for card in discard_pile 
                               if card.rank in [Rank.KING, Rank.QUEEN, Rank.JACK]]
        assert len(face_cards_in_discard) >= 2, "Face cards should be in discard pile after Six effect"
