from unittest.mock import MagicMock, patch

import pytest

from game.card import Card, Rank, Suit
from game.game_state import GameState
from tests.test_main.test_main_base import MainTestBase


class TestMainJack(MainTestBase):
    def generate_test_deck(self, p0_cards, p1_cards):
        """Generate a test deck with specific cards for each player."""
        deck = []
        # Add player 0's cards
        for card in p0_cards:
            deck.append(card)
        # Add player 1's cards
        for card in p1_cards:
            deck.append(card)
        # Add some random cards to make up the rest of the deck
        for suit in Suit:
            for rank in Rank:
                if rank not in [Rank.JACK, Rank.QUEEN, Rank.KING]:
                    card = Card(f"{len(deck)}", suit, rank)
                    if card not in deck:
                        deck.append(card)
        return deck

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_jack_on_opponent_point_card(
        self, mock_generate_cards, mock_input
    ):
        """Test playing a Jack on an opponent's point card through main.py."""
        # Create a mock logger
        mock_logger = MagicMock()

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
        self.mock_logger = mock_logger  # Store mock logger if needed later

        # Import and run main
        from main import main

        # Simpler approach: Patch GameState.__init__ within the Game initialization context
        with patch(
            "game.game.GameState.__init__",
            side_effect=lambda *args, **kwargs: GameState(
                *args, **{**kwargs, "logger": mock_logger}
            ),
        ):
            await main()

        # Get logger output
        log_output = self.get_logger_output(mock_logger)
        self.print_game_output(log_output)

        # Verify that the Jack was played on the opponent's point card
        # Assert based on logger output (GameState.print_state calls)
        self.assertIn(
            "Player 0: Score = 8, Target = 21", log_output
        )  # P0 score includes stolen 8C
        self.assertIn(
            "Field: [[Stolen from opponent] [Jack] Eight of Clubs]", log_output
        )  # P1 field shows stolen card

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("game.game.Game.generate_all_cards")
    async def test_cannot_play_jack_with_queen_on_field(
        self, mock_generate_cards, mock_input
    ):
        """Test that a Jack cannot be played if the opponent has a Queen on their field."""
        # Create a mock logger
        mock_logger = MagicMock()

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
            "0",  # P0: Draw card (action index 0 is Draw)
            "e",  # end game
            "n",  # Don't save game history
        ]
        self.setup_mock_input(mock_input, mock_inputs)
        self.mock_logger = mock_logger

        # Import and run main
        from main import main

        with patch(
            "game.game.GameState.__init__",
            side_effect=lambda *args, **kwargs: GameState(
                *args, **{**kwargs, "logger": mock_logger}
            ),
        ):
            await main()

        # Get logger output
        log_output = self.get_logger_output(mock_logger)
        self.print_game_output(log_output)

        # Verify that the illegal Jack action wasn't printed
        self.assertNotIn("Play Jack of Hearts as jack on Seven of Diamonds", log_output)
        # Verify the state after P1 plays 7D (before P0's turn where Jack is illegal)
        self.assertIn("Player 1: Score = 7, Target = 21", log_output)
        self.assertIn(
            "Player 1.*Field: [Queen of Clubs, Seven of Diamonds]",
            log_output,
            regex=True,
        )

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("game.game.Game.generate_all_cards")
    async def test_multiple_jacks_on_same_card(self, mock_generate_cards, mock_input):
        """Test that multiple jacks can be played on the same card."""
        # Create a mock logger
        mock_logger = MagicMock()

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
        self.mock_logger = mock_logger

        # Import and run main
        from main import main

        with patch(
            "game.game.GameState.__init__",
            side_effect=lambda *args, **kwargs: GameState(
                *args, **{**kwargs, "logger": mock_logger}
            ),
        ):
            await main()

        # Get logger output
        log_output = self.get_logger_output(mock_logger)
        self.print_game_output(log_output)

        # Assert based on logger output (GameState.print_state calls)
        # Check state after first jack
        self.assertIn("Player 0: Score = 3", log_output)
        self.assertIn(
            "Field: [[Stolen from opponent] [Jack] Three of Hearts]", log_output
        )
        # Check state after second jack
        self.assertIn("Player 1: Score = 3", log_output)
        self.assertIn("Field: [[Jack][Jack] Three of Hearts]", log_output)
        # Check state after third jack
        self.assertIn("Player 0: Score = 3", log_output)  # Score doesn't change
        self.assertIn(
            "Field: [[Stolen from opponent] [Jack][Jack][Jack] Three of Hearts]",
            log_output,
        )
        # Check state after fourth jack
        self.assertIn("Player 1: Score = 3", log_output)  # Score doesn't change
        self.assertIn("Field: [[Jack][Jack][Jack][Jack] Three of Hearts]", log_output)
