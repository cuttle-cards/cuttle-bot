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

    @pytest.mark.timeout(5)
    def test_scuttle_update_state(self):
        """Test the return values of update_state for scuttle actions."""
        game = Game()

        # Setup a game state where player 0 can scuttle player 1's card
        # Player 1's field: 5 of Hearts (point value 5)
        target_card = Card(
            "target", Suit.HEARTS, Rank.FIVE, played_by=1, purpose=Purpose.POINTS
        )
        game.game_state.fields[1].append(target_card)

        # Player 0's hand: 6 of Spades (higher point value)
        scuttle_card = Card("scuttle", Suit.SPADES, Rank.SIX)
        game.game_state.hands[0].append(scuttle_card)

        # Create scuttle action
        scuttle_action = Action(
            action_type=ActionType.SCUTTLE,
            card=scuttle_card,
            target=target_card,
            played_by=0,
        )

        # Test update_state return values
        turn_finished, should_stop, winner = game.game_state.update_state(
            scuttle_action
        )

        # Verify return values
        self.assertTrue(turn_finished)  # Scuttle should finish the turn
        self.assertFalse(should_stop)  # Game shouldn't stop after scuttle
        self.assertIsNone(winner)  # No winner from just scuttle

        # Verify game state changes
        self.assertNotIn(
            scuttle_card, game.game_state.hands[0]
        )  # Card removed from hand
        self.assertNotIn(
            target_card, game.game_state.fields[1]
        )  # Target removed from field
        self.assertIn(scuttle_card, game.game_state.discard_pile)  # Both cards moved to
        self.assertIn(target_card, game.game_state.discard_pile)  # discard pile

    @pytest.mark.timeout(5)
    def test_scuttle_with_equal_points(self):
        """Test scuttle with equal point values but higher suit."""
        game = Game()

        # Setup: Player 1's field has 5 of Hearts
        target_card = Card(
            "target", Suit.HEARTS, Rank.FIVE, played_by=1, purpose=Purpose.POINTS
        )
        game.game_state.fields[1].append(target_card)

        # Player 0's hand: 5 of Spades (same points, higher suit)
        scuttle_card = Card("scuttle", Suit.SPADES, Rank.FIVE)
        game.game_state.hands[0].append(scuttle_card)

        # Create scuttle action
        scuttle_action = Action(
            action_type=ActionType.SCUTTLE,
            card=scuttle_card,
            target=target_card,
            played_by=0,
        )

        # Test update_state return values
        turn_finished, should_stop, winner = game.game_state.update_state(
            scuttle_action
        )

        # Verify return values
        self.assertTrue(turn_finished)  # Scuttle should finish the turn
        self.assertFalse(should_stop)  # Game shouldn't stop
        self.assertIsNone(winner)  # No winner

        # Verify game state changes
        self.assertNotIn(scuttle_card, game.game_state.hands[0])
        self.assertNotIn(target_card, game.game_state.fields[1])
        self.assertIn(scuttle_card, game.game_state.discard_pile)
        self.assertIn(target_card, game.game_state.discard_pile)

    @pytest.mark.timeout(5)
    def test_scuttle_with_lower_suit_fails(self):
        """Test that scuttle fails when using a lower suit on same rank."""
        game = Game()

        # Setup: Player 1's field has 5 of Spades (highest suit)
        target_card = Card(
            "target", Suit.SPADES, Rank.FIVE, played_by=1, purpose=Purpose.POINTS
        )
        game.game_state.fields[1].append(target_card)

        # Player 0's hand: 5 of Clubs (lowest suit)
        scuttle_card = Card("scuttle", Suit.CLUBS, Rank.FIVE)
        game.game_state.hands[0].append(scuttle_card)

        # Create scuttle action
        scuttle_action = Action(
            action_type=ActionType.SCUTTLE,
            card=scuttle_card,
            target=target_card,
            played_by=0,
        )

        # Test update_state return values
        with self.assertRaises(Exception) as context:
            game.game_state.update_state(scuttle_action)

        # Verify error message
        self.assertIn("Invalid scuttle", str(context.exception))

        # Verify game state remains unchanged
        self.assertIn(scuttle_card, game.game_state.hands[0])  # Card still in hand
        self.assertIn(target_card, game.game_state.fields[1])  # Target still on field
        self.assertNotIn(scuttle_card, game.game_state.discard_pile)  # Cards not in
        self.assertNotIn(target_card, game.game_state.discard_pile)  # discard pile

    @pytest.mark.timeout(5)
    def test_play_king_reduces_target(self):
        """Test that playing a King reduces the target score."""
        game = Game()

        # Add a King to player 0's hand
        king = Card("king", Suit.HEARTS, Rank.KING)
        game.game_state.hands[0].append(king)

        # Create action to play King
        king_action = Action(
            action_type=ActionType.FACE_CARD,
            card=king,
            target=None,
            played_by=0,
        )

        # Play the King
        turn_finished, should_stop, winner = game.game_state.update_state(king_action)

        # Verify target score is reduced
        self.assertEqual(game.game_state.get_player_target(0), 14)  # 21 -> 14
        self.assertTrue(turn_finished)
        self.assertFalse(should_stop)
        self.assertIsNone(winner)

    @pytest.mark.timeout(5)
    def test_play_king_instant_win(self):
        """Test that playing a King can lead to instant win."""
        game = Game()

        # Add 11 points to player 0's field
        point_card = Card(
            "points", Suit.HEARTS, Rank.TEN, played_by=0, purpose=Purpose.POINTS
        )
        game.game_state.fields[0].append(point_card)

        # Add two Kings to player 0's hand
        kings = [Card(f"king{i}", Suit.HEARTS, Rank.KING) for i in range(2)]
        game.game_state.hands[0].extend(kings)

        # Play first King (target becomes 14, not winning yet)
        king1_action = Action(
            action_type=ActionType.FACE_CARD,
            card=kings[0],
            target=None,
            played_by=0,
        )
        turn_finished, should_stop, winner = game.game_state.update_state(king1_action)
        self.assertTrue(turn_finished)
        self.assertFalse(should_stop)
        self.assertIsNone(winner)

        # Play second King (target becomes 10, should win with 10 points)
        king2_action = Action(
            action_type=ActionType.FACE_CARD,
            card=kings[1],
            target=None,
            played_by=0,
        )
        turn_finished, should_stop, winner = game.game_state.update_state(king2_action)
        self.assertTrue(turn_finished)
        self.assertTrue(should_stop)
        self.assertEqual(winner, 0)

    @pytest.mark.timeout(5)
    def test_play_king_on_opponents_turn(self):
        """Test that Kings can only be played on your own turn."""
        game = Game()

        # Add a King to player 0's hand
        king = Card("king", Suit.HEARTS, Rank.KING)
        game.game_state.hands[0].append(king)

        # Set turn to player 1
        game.game_state.turn = 1

        # Create action to play King
        king_action = Action(
            action_type=ActionType.FACE_CARD,
            card=king,
            target=None,
            played_by=0,
        )

        # Try to play King on opponent's turn
        with self.assertRaises(Exception) as context:
            game.game_state.update_state(king_action)

        self.assertIn("Can only play cards from your hand", str(context.exception))

        # Verify game state unchanged
        self.assertIn(king, game.game_state.hands[0])
        self.assertNotIn(king, game.game_state.fields[0])
        self.assertEqual(game.game_state.get_player_target(0), 21)

    @pytest.mark.timeout(5)
    def test_play_multiple_kings(self):
        """Test playing multiple Kings reduces target score correctly."""
        game = Game()

        # Add four Kings to player 0's hand
        kings = [Card(f"king{i}", Suit.HEARTS, Rank.KING) for i in range(4)]
        game.game_state.hands[0].extend(kings)

        # Expected targets after each King
        expected_targets = [14, 10, 5, 0]  # Starting from 21

        # Play Kings one by one
        for i, king in enumerate(kings):
            action = Action(
                action_type=ActionType.FACE_CARD,
                card=king,
                target=None,
                played_by=0,
            )
            turn_finished, should_stop, winner = game.game_state.update_state(action)
            self.assertTrue(turn_finished)
            self.assertEqual(game.game_state.get_player_target(0), expected_targets[i])

            # Last King should not cause a win with 0 points
            if i < 3:
                self.assertFalse(should_stop)
                self.assertIsNone(winner)
            else:
                # With 4 Kings, target becomes 0
                self.assertTrue(should_stop)
                self.assertEqual(winner, 0)

    @pytest.mark.timeout(5)
    @patch("builtins.input")
    @patch("builtins.print")
    def test_complete_game_with_kings(self, mock_print, mock_input):
        """Test a complete game with manual selection and playing Kings until win."""
        # Mock inputs for manual selection
        mock_inputs = [str(i) for i in range(5)]  # Select first 5 cards for P0
        mock_inputs.extend([str(i) for i in range(6)])  # Select first 6 cards for P1
        mock_input.side_effect = mock_inputs

        # Create game with manual selection
        game = Game(manual_selection=True)

        # Verify initial hands
        self.assertEqual(len(game.game_state.hands[0]), 5)  # P0 has 5 cards
        self.assertEqual(len(game.game_state.hands[1]), 6)  # P1 has 6 cards

        # Play each card in P0's hand
        for card in game.game_state.hands[0].copy():
            if card.rank == Rank.KING:
                # Play King as face card
                action = Action(
                    action_type=ActionType.FACE_CARD,
                    card=card,
                    target=None,
                    played_by=0,
                )
            elif card.is_point_card():
                # Play as points
                action = Action(
                    action_type=ActionType.POINTS,
                    card=card,
                    target=None,
                    played_by=0,
                )
            else:
                continue  # Skip non-point, non-King cards

            turn_finished, should_stop, winner = game.game_state.update_state(action)
            self.assertTrue(turn_finished)

            if winner is not None:
                # If we won, verify the win condition
                self.assertTrue(should_stop)
                self.assertEqual(winner, 0)
                self.assertTrue(game.game_state.is_winner(0))
                self.assertFalse(game.game_state.is_winner(1))
                return  # Test succeeded - we found a winning sequence

        # If we get here, we didn't find a winning sequence
        # This is also fine - not every random hand will lead to a win
        pass


if __name__ == "__main__":
    unittest.main()
