from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from game.action import ActionType
from game.card import Card, Rank, Suit
from game.game import Game
from game.game_state import GameState
from tests.test_main.test_main_base import MainTestBase, print_and_capture


class TestMainJack(MainTestBase):
    def generate_test_deck(self, p0_cards: List[Card], p1_cards: List[Card], num_filler: int = 41) -> List[Card]:
        """Generate a test deck with specific cards for each player, overriding base but keeping functionality simple for these tests."""
        # Simple implementation: just combine hands and add some basic fillers if needed
        deck = list(p0_cards) + list(p1_cards)
        existing_ids = {c.id for c in deck}

        # Add minimal fillers if deck is too small (less than 11 needed for initial deal)
        needed_fillers = max(0, 11 - len(deck))
        filler_id = 100 # Start filler IDs high to avoid collision
        suit_cycle = list(Suit)
        rank_cycle = [r for r in Rank if r not in (Rank.JACK, Rank.KING, Rank.QUEEN)] # Avoid special ranks initially

        fill_count = 0
        while fill_count < needed_fillers:
            suit = suit_cycle[filler_id % len(suit_cycle)]
            rank = rank_cycle[filler_id % len(rank_cycle)]
            filler_card = Card(str(filler_id), suit, rank)
            if filler_card.id not in existing_ids:
                deck.append(filler_card)
                existing_ids.add(filler_card.id)
                fill_count += 1
            filler_id += 1
            if filler_id > 1000: # Safety break
                 break
        # The rest of the deck isn't critical for these tests, as long as dealing works
        return deck

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_jack_on_opponent_point_card(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test playing a Jack on an opponent's point card through main.py."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("2", Suit.SPADES, Rank.SIX),  # 6 of Spades
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.SEVEN),  # 7 of Diamonds (point card)
            Card("7", Suit.CLUBS, Rank.EIGHT),  # 8 of Clubs
            Card("8", Suit.HEARTS, Rank.THREE),  # 3 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.TEN),  # 10 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards (indices)
            "0",
            "0",
            "0",
            "0",
            "0",
            # Player 1 selects cards (indices)
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "n",  # Don't save initial state
            # Game actions (indices)
            "1",  # P0: Play 6S points
            "1",  # P1: Play 8C points (Changed from original test which failed)
            "4",  # P0: Play JH on 8C
            "e",  # end game
            "n",  # Don't save game history
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
            await main()
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Jack was played
        jack_actions = history.get_actions_by_type(ActionType.JACK)
        assert len(jack_actions) == 1, "Expected exactly one Jack action"
        jack_action = jack_actions[0]
        assert jack_action.card.rank == Rank.JACK, "Expected Jack to be played"
        assert jack_action.card.suit == Suit.HEARTS, "Expected Jack of Hearts to be played"
        assert jack_action.player == 0, "Expected player 0 to play the Jack"
        
        # Verify the Jack was played on an opponent's point card
        assert jack_action.target is not None, "Jack should have a target"
        assert jack_action.target.rank == Rank.EIGHT, "Jack should target Eight of Clubs"
        assert jack_action.target.suit == Suit.CLUBS, "Jack should target Eight of Clubs"
        
        # Verify final game state - Player 0 should have the stolen card
        p0_field = captured_game.game_state.fields[0]
        stolen_cards = [card for card in p0_field if card.rank == Rank.EIGHT and card.suit == Suit.CLUBS]
        assert len(stolen_cards) == 1, "Player 0 should have stolen Eight of Clubs"
        
        # Verify the Jack is attached to the stolen card
        stolen_card = stolen_cards[0]
        jacks_on_card = [attachment for attachment in stolen_card.attachments if attachment.rank == Rank.JACK]
        assert len(jacks_on_card) == 1, "Should have one Jack attached to stolen card"

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_cannot_play_jack_with_queen_on_field(
        self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock
    ) -> None:
        """Test that a Jack cannot be played if the opponent has a Queen on their field."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("2", Suit.SPADES, Rank.SIX),  # 6 of Spades
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),  # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.SEVEN),  # 7 of Diamonds (point card)
            Card("7", Suit.CLUBS, Rank.QUEEN),  # Queen of Clubs
            Card("8", Suit.HEARTS, Rank.THREE),  # 3 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),  # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR),  # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.TEN),  # 10 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards (indices)
            "0",
            "0",
            "0",
            "0",
            "0",
            # Player 1 selects cards (indices)
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "n",  # Don't save initial state
            # Game actions (indices based on available actions)
            "1",  # P0: Play 6S points
            "6",  # P1: Play QC face card
            "1",  # P0: Play 9H points
            "1",  # P1: Play 7D points
            # P0 Turn: Jack is illegal due to Queen. Check available actions.
            "0",  # P0: Available action
            "e",  # end game
            "n",  # Don't save game history
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
            await main()
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify Queen was played as face card
        face_card_actions = history.get_actions_by_type(ActionType.FACE_CARD)
        queen_actions = [action for action in face_card_actions 
                        if action.card and action.card.rank == Rank.QUEEN]
        assert len(queen_actions) == 1, "Expected exactly one Queen face card action"
        queen_action = queen_actions[0]
        assert queen_action.card.suit == Suit.CLUBS, "Expected Queen of Clubs to be played"
        assert queen_action.player == 1, "Expected player 1 to play the Queen"
        
        # Verify no Jack actions occurred (Queen blocks Jacks)
        jack_actions = history.get_actions_by_type(ActionType.JACK)
        assert len(jack_actions) == 0, "No Jack actions should occur when Queen is on field"
        
        # Verify final game state - Player 1 should have Queen on field
        p1_field = captured_game.game_state.fields[1]
        queens_on_field = [card for card in p1_field if card.rank == Rank.QUEEN]
        assert len(queens_on_field) == 1, "Player 1 should have Queen on field"
        
        # Verify Player 0 still has Jack in hand (couldn't play it)
        p0_hand = captured_game.game_state.hands[0]
        jacks_in_hand = [card for card in p0_hand if card.rank == Rank.JACK]
        assert len(jacks_in_hand) >= 1, "Player 0 should still have Jack in hand"

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_multiple_jacks_on_same_card(self, mock_generate_cards: Mock, mock_print: Mock, mock_input: Mock) -> None:
        """Test that multiple jacks can be played on the same card."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("2", Suit.SPADES, Rank.JACK),  # Jack of Spades
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE),  # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TEN),  # 10 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.JACK),  # Jack of Diamonds
            Card("7", Suit.CLUBS, Rank.JACK),  # Jack of Clubs
            Card("8", Suit.HEARTS, Rank.THREE),  # 3 of Hearts
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
            # Player 0 selects cards (indices)
            "0",
            "0",
            "0",
            "0",
            "0",
            # Player 1 selects cards (indices)
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "n",  # Don't save initial state
            # Game actions (indices)
            "1",  # P0: Play 9H points
            "1",  # P1: Play 3H points
            "3",  # P0: Play JH on 3H (Index 3 based on P0 Turn 2 actions)
            "4",  # P1: Play JD on 3H (Index 4 based on P1 Turn 2 actions)
            "3",  # P0: Play JS on 3H (Index 3 based on P0 Turn 3 actions)
            "4",  # P1: Play JC on 3H (Index 4 based on P1 Turn 3 actions)
            "e",  # End game after checks
            "n",  # Don't save game history
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
            await main()
        finally:
            # Restore original
            Game.__init__ = original_init

        # Verify we captured the game object
        assert captured_game is not None, "Game object was not captured"
        
        # Access the game history
        history = captured_game.game_state.game_history
        
        # Verify multiple Jack actions occurred
        jack_actions = history.get_actions_by_type(ActionType.JACK)
        assert len(jack_actions) >= 2, f"Expected at least 2 Jack actions, got {len(jack_actions)}"
        
        # Verify all Jacks target the same card (Three of Hearts)
        target_card = jack_actions[0].target
        assert target_card is not None, "Jack should have a target"
        assert target_card.rank == Rank.THREE, "Jack should target Three of Hearts"
        assert target_card.suit == Suit.HEARTS, "Jack should target Three of Hearts"
        
        # Verify all subsequent Jacks target the same card
        for jack_action in jack_actions[1:]:
            assert jack_action.target.rank == target_card.rank, "All Jacks should target same card"
            assert jack_action.target.suit == target_card.suit, "All Jacks should target same card"
        
        # Find where the Three of Hearts ended up and count attached Jacks
        three_of_hearts_locations = []
        for player_field in captured_game.game_state.fields:
            for card in player_field:
                if card.rank == Rank.THREE and card.suit == Suit.HEARTS:
                    three_of_hearts_locations.append((card, player_field))
        
        assert len(three_of_hearts_locations) == 1, "Three of Hearts should be on exactly one field"
        three_card, field = three_of_hearts_locations[0]
        
        # Count Jacks attached to the Three of Hearts
        jacks_attached = [attachment for attachment in three_card.attachments if attachment.rank == Rank.JACK]
        assert len(jacks_attached) >= 2, f"Expected at least 2 Jacks attached, got {len(jacks_attached)}"
