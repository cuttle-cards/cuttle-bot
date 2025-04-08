from unittest.mock import patch, MagicMock
import pytest
from game.card import Card, Suit, Rank, Purpose
from tests.test_main.test_main_base import MainTestBase, print_and_capture

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
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_play_jack_on_opponent_point_card(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test playing a Jack on an opponent's point card through main.py."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("2", Suit.SPADES, Rank.SIX),   # 6 of Spades
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE), # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),    # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.SEVEN), # 7 of Diamonds (point card)
            Card("7", Suit.CLUBS, Rank.EIGHT),    # 8 of Clubs
            Card("8", Suit.HEARTS, Rank.THREE),   # 3 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),    # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR), # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.TEN),     # 10 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 0
            # Player 1 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 1
            "n",  # Don't save initial state
            # Game actions
            "Two of Clubs as points",
            "Seven of Diamonds as points",
            "Jack of Hearts as jack on seven of diamonds",
            "e",  # end game
            "n",  # Don't save game history
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        # Verify that the Jack was played on the opponent's point card
        self.assertIn("Player 0's field: [Two of Clubs, [Stolen from opponent] [Jack] Seven of Diamonds]", log_output)

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_cannot_play_jack_with_queen_on_field(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test that a Jack cannot be played if the opponent has a Queen on their field."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("2", Suit.SPADES, Rank.SIX),   # 6 of Spades
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE), # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TWO),    # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.SEVEN), # 7 of Diamonds (point card)
            Card("7", Suit.CLUBS, Rank.QUEEN),    # Queen of Clubs
            Card("8", Suit.HEARTS, Rank.THREE),   # 3 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),    # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR), # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.TEN),     # 10 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 0
            # Player 1 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 1
            "n",  # Don't save initial state
            # Game actions
            "Six of Spades as points",  # Player 0 plays Six of Spades as points
            "Queen of Clubs as face card",  # Player 1 plays Queen of Clubs as face card
            "Nine of Hearts as points",  # Player 1 scuttles Nine of Hearts
            "Seven of Diamonds as points",  # Player 1 plays Seven of Diamonds as points
            # Player 0 tries to play Jack of Hearts on Seven of Diamonds. This is not in the legal actions, so it should not be in the log output and Draw card should be the final selection
            "Jack of Hearts as jack on seven of diamonds",  
            "e",  # end game
            "n",  # Don't save game history
        ]
        self.setup_mock_input(mock_input, mock_inputs)

        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        # verify that "Jack of Hearts as jack on seven of diamonds" is not in the last few lines of the log output
        self.assertNotIn("Jack of Hearts as jack on seven of diamonds", log_output[-30:])

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    @patch("game.game.Game.generate_all_cards")
    async def test_multiple_jacks_on_same_card(
        self, mock_generate_cards, mock_print, mock_input
    ):
        """Test that multiple jacks can be played on the same card."""
        # Set up print mock to both capture and display
        mock_print.side_effect = print_and_capture

        # Create test deck with specific cards
        
        p0_cards = [
            Card("1", Suit.HEARTS, Rank.JACK),  # Jack of Hearts
            Card("2", Suit.SPADES, Rank.JACK),   # 6 of Spades
            Card("3", Suit.HEARTS, Rank.NINE),  # 9 of Hearts
            Card("4", Suit.DIAMONDS, Rank.FIVE), # 5 of Diamonds
            Card("5", Suit.CLUBS, Rank.TEN),    # 2 of Clubs
        ]
        p1_cards = [
            Card("6", Suit.DIAMONDS, Rank.JACK), # 7 of Diamonds (point card)
            Card("7", Suit.CLUBS, Rank.JACK),    # Queen of Clubs
            Card("8", Suit.HEARTS, Rank.THREE),   # 3 of Hearts
            Card("9", Suit.SPADES, Rank.FIVE),    # 5 of Spades
            Card("10", Suit.DIAMONDS, Rank.FOUR), # 4 of Diamonds
            Card("11", Suit.CLUBS, Rank.TWO),     # 10 of Clubs
        ]
        test_deck = self.generate_test_deck(p0_cards, p1_cards)
        mock_generate_cards.return_value = test_deck

        # Mock sequence of inputs for the entire game
        mock_inputs = [
            "n",  # Don't load saved game
            "y",  # Use manual selection
            # Player 0 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 0
            # Player 1 selects cards
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",  # Select all cards for Player 1
            "n",  # Don't save initial state
            # Game actions
            "Ten of Clubs as points",
            "Jack of Clubs as jack on Ten of clubs",
            "Play Jack of Hearts as jack on [Stolen from opponent] [Jack] Ten of Clubs",
            "Play Jack of Diamonds as jack on [Jack][Jack] Ten of Clubs",
            "Play Jack of Spades as jack on [Stolen from opponent] [Jack][Jack][Jack] Ten of Clubs",
            "e",  # end game
            "n",  # Don't save game history
        ]
        self.setup_mock_input(mock_input, mock_inputs)
        
        # Import and run main
        from main import main

        await main()

        # Get all logged output
        log_output = self.get_log_output()
        self.print_game_output(log_output)

        self.assertIn("Player 1's field: [[Stolen from opponent] [Jack] Ten of Clubs]", log_output)
        self.assertIn("Player 0's field: [[Jack][Jack] Ten of Clubs]", log_output)
        self.assertIn("Player 1's field: [[Stolen from opponent] [Jack][Jack][Jack] Ten of Clubs]", log_output)
        self.assertIn("Player 0's field: [[Jack][Jack][Jack][Jack] Ten of Clubs]", log_output)

