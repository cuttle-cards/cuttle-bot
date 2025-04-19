import unittest
from unittest.mock import patch

import pytest

from game.action import Action, ActionType
from game.card import Card, Purpose, Rank, Suit
from game.game import Game
from game.game_state import GameState


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

        # Pass the mock_print function as the logger
        game = Game(manual_selection=True, logger=mock_print)

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

        # Pass the mock_print function as the logger
        game = Game(manual_selection=True, logger=mock_print)

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

        # Pass the mock_print function as the logger
        game = Game(manual_selection=True, logger=mock_print)

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
        scuttle_card = Card("scuttle", Suit.SPADES, Rank.SIX, played_by=0, purpose=Purpose.SCUTTLE)
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
        scuttle_card = Card("scuttle", Suit.SPADES, Rank.FIVE, played_by=0, purpose=Purpose.SCUTTLE)
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

    @patch("builtins.input")
    @patch("builtins.print")
    def test_complete_game_with_kings(self, mock_print, mock_input):
        """Test a complete game scenario ending with King win condition."""
        # Mock inputs: P0 gets Kings, P1 gets points, P0 wins
        mock_inputs = [
            "0",
            "1",
            "2",
            "3",
            "4",  # P0 selects Kings + Ace
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",  # P1 selects high points
            # Game actions (mocked, not used as test deck is provided)
        ]
        mock_input.side_effect = mock_inputs

        # Test deck: P0 gets 4 Kings + Ace, P1 gets points
        test_deck = [
            Card("KH", Suit.HEARTS, Rank.KING),
            Card("KD", Suit.DIAMONDS, Rank.KING),
            Card("KS", Suit.SPADES, Rank.KING),
            Card("KC", Suit.CLUBS, Rank.KING),
            Card("AH", Suit.HEARTS, Rank.ACE),  # P0 hand
            Card("10H", Suit.HEARTS, Rank.TEN),
            Card("10D", Suit.DIAMONDS, Rank.TEN),
            Card("10S", Suit.SPADES, Rank.TEN),
            Card("9H", Suit.HEARTS, Rank.NINE),
            Card("8H", Suit.HEARTS, Rank.EIGHT),
            Card("7H", Suit.HEARTS, Rank.SEVEN),  # P1 hand
        ] + [Card(str(i), Suit.CLUBS, Rank.TWO) for i in range(41)]  # Filler

        # Pass the mock_print function as the logger
        game = Game(test_deck=test_deck, logger=mock_print)

        # Simulate game play actions (assuming direct state manipulation for brevity)
        # Player 0 plays 4 Kings
        for i in range(4):
            king = game.game_state.hands[0].pop(0)
            game.game_state.fields[0].append(king)

        # Check win condition (target 0)
        self.assertEqual(game.game_state.get_player_target(0), 0)
        self.assertTrue(game.game_state.is_winner(0))
        self.assertEqual(game.game_state.winner(), 0)

        # Check final state print
        game.game_state.print_state()
        mock_print.assert_called()

    def test_play_jack_action(self):
        """Test playing a Jack action to steal a point card from opponent."""
        # Set up initial state with point cards on opponent's field
        hands = [
            [Card("1", Suit.HEARTS, Rank.JACK)],  # Player 0's hand with Jack
            [Card("2", Suit.DIAMONDS, Rank.TWO)],  # Player 1's hand
        ]
        fields = [
            [],  # Player 0's field (empty)
            [  # Player 1's field with point cards
                Card("3", Suit.SPADES, Rank.SEVEN, played_by=1, purpose=Purpose.POINTS),
                Card("4", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS),
            ],
        ]
        deck = []
        discard = []

        game_state = GameState(hands, fields, deck, discard)

        # Verify initial scores
        self.assertEqual(game_state.get_player_score(0), 0)
        self.assertEqual(game_state.get_player_score(1), 16)  # 7 + 9 = 16

        # Create Jack action
        jack_card = hands[0][0]
        target_card = fields[1][0]  # Seven of Spades
        jack_action = Action(action_type=ActionType.JACK, card=jack_card, target=target_card, played_by=0)

        # Apply the action
        turn_finished, should_stop, winner = game_state.update_state(jack_action)

        # Verify action was successful
        self.assertTrue(turn_finished)
        self.assertFalse(should_stop)
        self.assertIsNone(winner)

        # Verify Jack is removed from hand
        self.assertNotIn(jack_card, game_state.hands[0])

        # Verify Jack is attached to the target card
        self.assertIn(jack_card, target_card.attachments)
        self.assertEqual(len(target_card.attachments), 1)
        for card in fields[1]:
            print(card, card.attachments)
        self.assertEqual(
            len(fields[1][1].attachments), 0
        )  # no attachments on the other point card

        # Verify the point card is now stolen (counts for player 0)
        self.assertTrue(target_card.is_stolen())

        # Verify scores are updated correctly
        self.assertEqual(
            game_state.get_player_score(0), 7
        )  # Stolen card counts for player 0
        self.assertEqual(
            game_state.get_player_score(1), 9
        )  # Only the second card counts for player 1

    def test_play_jack_action_with_queen_on_field(self):
        """Test that Jack action cannot be played if opponent has a Queen on their field."""
        # Set up initial state with a Queen on opponent's field
        hands = [
            [Card("1", Suit.HEARTS, Rank.JACK)],  # Player 0's hand with Jack
            [Card("2", Suit.DIAMONDS, Rank.TWO)],  # Player 1's hand
        ]
        fields = [
            [],  # Player 0's field (empty)
            [  # Player 1's field with Queen and point card
                Card(
                    "3", Suit.SPADES, Rank.QUEEN, played_by=1, purpose=Purpose.FACE_CARD
                ),
                Card("4", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS),
            ],
        ]
        deck = []
        discard = []

        game_state = GameState(hands, fields, deck, discard)

        # Create Jack action
        jack_card = hands[0][0]
        target_card = fields[1][1]  # Nine of Hearts
        jack_action = Action(action_type=ActionType.JACK, card=jack_card, target=target_card, played_by=0)

        # Try to apply the action
        with self.assertRaises(Exception) as context:
            game_state.update_state(jack_action)

        # Verify error message
        self.assertIn(
            "Cannot play jack as face card if opponent has a queen on their field",
            str(context.exception),
        )

        # Verify game state unchanged
        self.assertIn(jack_card, game_state.hands[0])
        self.assertEqual(len(target_card.attachments), 0)
        self.assertFalse(target_card.is_stolen())


if __name__ == "__main__":
    unittest.main()
