import unittest
from unittest.mock import patch
import pytest
from game.game import Game
from game.card import Card, Suit, Rank, Purpose
from game.action import Action, ActionType


class TestGame(unittest.TestCase):
    @pytest.mark.timeout(5)
    def test_initialize_with_random_hands(self):
        """Test that random initialization creates valid hands."""
        game = Game(manual_selection=False)

        # Check hand sizes
        self.assertEqual(
            len(game.game_state.hands[0]), 5
        )  # Player 0 should have 5 cards
        self.assertEqual(
            len(game.game_state.hands[1]), 6
        )  # Player 1 should have 6 cards

        # Check deck size (52 - 11 = 41 cards should remain)
        self.assertEqual(len(game.game_state.deck), 41)

        # Check that all cards are unique
        all_cards = (
            game.game_state.hands[0] + game.game_state.hands[1] + game.game_state.deck
        )
        unique_cards = set(str(card) for card in all_cards)
        self.assertEqual(len(all_cards), len(unique_cards))
        self.assertEqual(len(all_cards), 52)  # Total number of cards

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    def test_manual_selection_full_hands(self, mock_print, mock_input):
        """Test manual selection when players select full hands."""
        # Mock inputs: Player 0 selects 5 cards, Player 1 selects 6 cards
        mock_inputs = [
            "0",
            "1",
            "2",
            "3",
            "4",  # Player 0's selections
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",  # Player 1's selections
        ]
        mock_input.side_effect = mock_inputs

        game = Game(manual_selection=True)

        # Check hand sizes
        self.assertEqual(len(game.game_state.hands[0]), 5)
        self.assertEqual(len(game.game_state.hands[1]), 6)

        # Check that selected cards are unique
        all_cards = (
            game.game_state.hands[0] + game.game_state.hands[1] + game.game_state.deck
        )
        unique_cards = set(str(card) for card in all_cards)
        self.assertEqual(len(all_cards), len(unique_cards))
        self.assertEqual(len(all_cards), 52)

        # Print state (now with mocked print function)
        game.game_state.print_state()

        # Verify print was called
        self.assertTrue(mock_print.called)

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    def test_manual_selection_early_done(self, mock_print, mock_input):
        """Test manual selection when players finish early with 'done'."""
        # Mock inputs: Player 0 selects 3 cards and done, Player 1 selects 4 cards and done
        mock_inputs = [
            "0",
            "1",
            "2",
            "done",  # Player 0's selections
            "0",
            "1",
            "2",
            "3",
            "done",  # Player 1's selections
        ]
        mock_input.side_effect = mock_inputs

        game = Game(manual_selection=True)

        # Check hand sizes (should still be full despite early done)
        self.assertEqual(len(game.game_state.hands[0]), 5)
        self.assertEqual(len(game.game_state.hands[1]), 6)

        # Check that all cards are unique
        all_cards = (
            game.game_state.hands[0] + game.game_state.hands[1] + game.game_state.deck
        )
        unique_cards = set(str(card) for card in all_cards)
        self.assertEqual(len(all_cards), len(unique_cards))
        self.assertEqual(len(all_cards), 52)

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    def test_manual_selection_invalid_inputs(self, mock_print, mock_input):
        """Test manual selection with invalid inputs before valid ones."""
        # Mock inputs: Include invalid inputs that should be handled
        mock_inputs = [
            "invalid",
            "-1",
            "999",
            "0",
            "1",
            "done",  # Player 0's selections (with invalid inputs)
            "abc",
            "1",
            "2",
            "done",  # Player 1's selections (with invalid input)
        ]
        mock_input.side_effect = mock_inputs

        game = Game(manual_selection=True)

        # Check hand sizes
        self.assertEqual(len(game.game_state.hands[0]), 5)
        self.assertEqual(len(game.game_state.hands[1]), 6)

        # Check that all cards are unique
        all_cards = (
            game.game_state.hands[0] + game.game_state.hands[1] + game.game_state.deck
        )
        unique_cards = set(str(card) for card in all_cards)
        self.assertEqual(len(all_cards), len(unique_cards))
        self.assertEqual(len(all_cards), 52)

    @pytest.mark.timeout(5)
    def test_generate_all_cards(self):
        """Test that generate_all_cards creates a complete deck."""
        game = Game()
        cards = game.generate_all_cards()

        # Check total number of cards
        self.assertEqual(len(cards), 52)

        # Check that all suits and ranks are represented
        suits = set(card.suit for card in cards)
        ranks = set(card.rank for card in cards)
        self.assertEqual(len(suits), 4)  # 4 suits
        self.assertEqual(len(ranks), 13)  # 13 ranks

        # Check uniqueness
        unique_cards = set(str(card) for card in cards)
        self.assertEqual(len(cards), len(unique_cards))

    @pytest.mark.timeout(5)
    def test_fill_remaining_slots(self):
        """Test that fill_remaining_slots correctly fills partial hands."""
        game = Game()

        # Create partial hands
        hands = [
            [
                Card("1", Suit.HEARTS, Rank.ACE),
                Card("2", Suit.HEARTS, Rank.TWO),
            ],  # 2 cards
            [Card("3", Suit.HEARTS, Rank.THREE)],  # 1 card
        ]

        # Create available cards (excluding the ones in hands)
        all_cards = game.generate_all_cards()
        available_cards = {
            str(card): card
            for card in all_cards
            if str(card) not in [str(c) for h in hands for c in h]
        }

        # Fill remaining slots
        game.fill_remaining_slots(hands, available_cards)

        # Check hand sizes
        self.assertEqual(len(hands[0]), 5)  # Should have added 3 cards
        self.assertEqual(len(hands[1]), 6)  # Should have added 5 cards

        # Check that all cards are unique
        all_hand_cards = hands[0] + hands[1]
        unique_cards = set(str(card) for card in all_hand_cards)
        self.assertEqual(len(all_hand_cards), len(unique_cards))

    @pytest.mark.timeout(5)
    def test_save_load_game(self):
        """Test saving and loading game state."""
        # Create a game with known state
        game = Game()
        initial_hands = [
            [card for card in game.game_state.hands[0]],
            [card for card in game.game_state.hands[1]],
        ]
        initial_deck = [card for card in game.game_state.deck]

        # Save the game
        test_file = "test_save.json"
        game.save_game(test_file)

        # Create a new game by loading the saved state
        loaded_game = Game(load_game=test_file)

        # Verify the loaded state matches the original
        for player in range(2):
            self.assertEqual(
                len(loaded_game.game_state.hands[player]), len(initial_hands[player])
            )
            for card1, card2 in zip(
                loaded_game.game_state.hands[player], initial_hands[player]
            ):
                self.assertEqual(str(card1), str(card2))

        self.assertEqual(len(loaded_game.game_state.deck), len(initial_deck))
        for card1, card2 in zip(loaded_game.game_state.deck, initial_deck):
            self.assertEqual(str(card1), str(card2))

        # Clean up
        import os

        save_path = os.path.join(Game.SAVE_DIR, test_file)
        if os.path.exists(save_path):
            os.remove(save_path)

    @pytest.mark.timeout(5)
    def test_list_saved_games(self):
        """Test listing saved games."""
        # Create some test save files
        test_files = ["test1.json", "test2.json", "test3.json"]
        game = Game()

        for filename in test_files:
            game.save_game(filename)

        # Get list of saved games
        saved_games = Game.list_saved_games()

        # Verify all test files are in the list
        for test_file in test_files:
            self.assertIn(test_file, saved_games)

        # Clean up
        import os

        for filename in test_files:
            save_path = os.path.join(Game.SAVE_DIR, filename)
            if os.path.exists(save_path):
                os.remove(save_path)

    @pytest.mark.timeout(5)
    def test_load_nonexistent_game(self):
        """Test loading a non-existent save file."""
        with self.assertRaises(FileNotFoundError):
            Game(load_game="nonexistent_save.json")
if __name__ == "__main__":
    unittest.main()
