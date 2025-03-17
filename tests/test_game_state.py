import unittest
from game.card import Card, Purpose, Suit, Rank
from game.game_state import GameState
from unittest.mock import patch


class TestGameState(unittest.TestCase):

    def setUp(self):
        # Create a sample deck and hands for testing
        self.deck = [Card(str(i), Suit.CLUBS, Rank.ACE) for i in range(10)]
        self.hands = [
            [Card(str(i), Suit.HEARTS, Rank.TWO) for i in range(5)],
            [Card(str(i), Suit.SPADES, Rank.THREE) for i in range(6)],
        ]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

    def test_initial_state(self):
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.hands, self.hands)
        self.assertEqual(self.game_state.fields, self.fields)
        self.assertEqual(self.game_state.deck, self.deck)
        self.assertEqual(self.game_state.discard_pile, self.discard_pile)

    def test_next_turn(self):
        self.game_state.next_turn()
        self.assertEqual(self.game_state.turn, 1)
        self.game_state.next_turn()
        self.assertEqual(self.game_state.turn, 0)

    def test_get_player_score(self):
        self.assertEqual(self.game_state.get_player_score(0), 0)
        self.assertEqual(self.game_state.get_player_score(1), 0)

    def test_get_player_target(self):
        self.assertEqual(self.game_state.get_player_target(0), 21)
        self.assertEqual(self.game_state.get_player_target(1), 21)
        self.fields[0].append(Card("", Suit.HEARTS, Rank.KING, played_by=0))
        self.assertEqual(self.game_state.get_player_target(0), 14)
        self.fields[0].append(Card("", Suit.CLUBS, Rank.KING, played_by=0))
        self.assertEqual(self.game_state.get_player_target(0), 10)
        self.fields[0].append(Card("", Suit.DIAMONDS, Rank.KING, played_by=0))
        self.assertEqual(self.game_state.get_player_target(0), 5)
        self.fields[0].append(Card("", Suit.SPADES, Rank.KING, played_by=0))
        self.assertEqual(self.game_state.get_player_target(0), 0)

    def test_is_winner(self):
        self.assertFalse(self.game_state.is_winner(0))
        self.assertFalse(self.game_state.is_winner(1))
        p0_new_cards = [
            Card("", Suit.HEARTS, Rank.KING, played_by=0),
            Card("", Suit.CLUBS, Rank.KING, played_by=0),
            Card("", Suit.DIAMONDS, Rank.TEN, purpose=Purpose.POINTS, played_by=0),
        ]
        self.fields[0].extend(p0_new_cards)
        self.assertEqual(self.game_state.get_player_score(0), 10)
        self.assertEqual(self.game_state.get_player_target(0), 10)
        self.assertTrue(self.game_state.is_winner(0))
        self.assertFalse(self.game_state.is_winner(1))

    def test_winner(self):
        self.assertIsNone(self.game_state.winner())

    def test_is_stalemate(self):
        self.assertFalse(self.game_state.is_stalemate())

    def test_draw_card(self):
        self.game_state.draw_card()
        self.assertEqual(len(self.game_state.hands[0]), 6)
        self.assertEqual(len(self.game_state.deck), 9)
        self.game_state.draw_card()
        self.assertEqual(len(self.game_state.hands[0]), 7)
        self.assertEqual(len(self.game_state.deck), 8)
        self.game_state.draw_card()
        self.assertEqual(len(self.game_state.hands[0]), 8)
        self.assertEqual(len(self.game_state.deck), 7)

        with self.assertRaises(Exception):
            self.game_state.draw_card()
            self.assertEqual(len(self.game_state.hands[0]), 8)
            self.assertEqual(len(self.game_state.deck), 7)

    def test_play_points(self):
        card = self.hands[0][0]
        self.game_state.play_points(card)
        self.assertIn(card, self.game_state.fields[0])
        self.assertNotIn(card, self.game_state.hands[0])

    def test_scuttle(self):
        card = self.hands[0][0]
        target = Card("target", Suit.CLUBS, Rank.TWO, played_by=1)
        self.game_state.fields[1].append(target)
        self.game_state.scuttle(card, target)
        self.assertIn(card, self.game_state.discard_pile)
        self.assertIn(target, self.game_state.discard_pile)
        self.assertNotIn(target, self.game_state.fields[1])

    def test_play_one_off(self):
        counter_card = Card(
            "counter", Suit.HEARTS, Rank.TWO, played_by=1, purpose=Purpose.COUNTER
        )

        self.hands[1].append(counter_card)

        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        finished, played_by = self.game_state.play_one_off(
            1, card, countered_with=counter_card
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)
        self.assertNotEqual(self.game_state.turn, played_by)

        counter_card_0 = Card(
            "counter2", Suit.DIAMONDS, Rank.TWO, played_by=0, purpose=Purpose.COUNTER
        )
        self.hands[0].append(counter_card_0)

        finished, played_by = self.game_state.play_one_off(
            0, card, countered_with=counter_card_0
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 0)

        finished, played_by = self.game_state.play_one_off(1, card, last_resolved_by=1)
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertIn(card, self.game_state.discard_pile)
        self.assertNotIn(card, self.game_state.hands[0])

    def test_play_five_one_off(self):
        self.deck = [
            Card("001", Suit.CLUBS, Rank.ACE),
            Card("002", Suit.CLUBS, Rank.TWO),
        ]
        self.hands = [
            [Card("003", Suit.HEARTS, Rank.FIVE)],
            [Card("004", Suit.SPADES, Rank.SIX)],
        ]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # play FIVE as ONE_OFF
        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 0)
        self.assertEqual(self.game_state.current_action_player, 0)
        self.game_state.next_player()
        self.assertEqual(self.game_state.current_action_player, 1)

        # resolve
        finished, played_by = self.game_state.play_one_off(
            1, card, None, last_resolved_by=self.game_state.current_action_player
        )
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 1)
        self.assertEqual(len(self.game_state.hands[0]), 2)

        self.game_state.next_turn()
        self.assertEqual(self.game_state.turn, 1)

    def test_play_five_one_off_with_eight_cards(self):
        self.deck = [
            Card("001", Suit.CLUBS, Rank.ACE),
            Card("002", Suit.CLUBS, Rank.TWO),
        ]
        self.hands = [
            [
                Card("003", Suit.HEARTS, Rank.FIVE),
                Card("004", Suit.HEARTS, Rank.ACE),
                Card("005", Suit.HEARTS, Rank.TWO),
                Card("006", Suit.HEARTS, Rank.THREE),
                Card("007", Suit.HEARTS, Rank.FOUR),
                Card("008", Suit.HEARTS, Rank.SIX),
                Card("009", Suit.HEARTS, Rank.SEVEN),
            ],
            [],
        ]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 0)
        self.assertEqual(self.game_state.current_action_player, 0)
        self.game_state.next_player()
        self.assertEqual(self.game_state.current_action_player, 1)

        finished, played_by = self.game_state.play_one_off(
            1, card, None, last_resolved_by=self.game_state.current_action_player
        )
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 1)

        self.assertEqual(len(self.game_state.hands[0]), 8)

        self.deck = [
            Card("001", Suit.CLUBS, Rank.ACE),
            Card("002", Suit.CLUBS, Rank.TWO),
        ]
        self.hands = [
            [
                Card("003", Suit.HEARTS, Rank.FIVE),
                Card("004", Suit.HEARTS, Rank.ACE),
                Card("005", Suit.HEARTS, Rank.TWO),
                Card("006", Suit.HEARTS, Rank.THREE),
                Card("007", Suit.HEARTS, Rank.FOUR),
                Card("008", Suit.HEARTS, Rank.SIX),
                Card("009", Suit.HEARTS, Rank.SEVEN),
                Card("010", Suit.HEARTS, Rank.EIGHT),
            ],
            [],
        ]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 0)
        self.assertEqual(self.game_state.current_action_player, 0)
        self.game_state.next_player()
        self.assertEqual(self.game_state.current_action_player, 1)

        finished, played_by = self.game_state.play_one_off(
            1, card, None, last_resolved_by=self.game_state.current_action_player
        )
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 1)

        self.assertEqual(len(self.game_state.hands[0]), 8)

    def test_play_ace_one_off(self):
        """Test playing an Ace as a one-off to destroy all point cards."""
        # Setup initial game state with point cards on both players' fields
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        self.hands = [
            [Card("002", Suit.HEARTS, Rank.ACE)],  # Player 0's hand with ACE
            [Card("003", Suit.SPADES, Rank.TWO)],  # Player 1's hand
        ]
        # Add point cards to both players' fields
        self.fields = [
            [
                Card(
                    "004",
                    Suit.DIAMONDS,
                    Rank.SEVEN,
                    played_by=0,
                    purpose=Purpose.POINTS,
                ),
                Card("005", Suit.CLUBS, Rank.FOUR, played_by=0, purpose=Purpose.POINTS),
            ],
            [
                Card(
                    "006", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS
                ),
                Card(
                    "007", Suit.SPADES, Rank.THREE, played_by=1, purpose=Purpose.POINTS
                ),
            ],
        ]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Play ACE as one-off
        ace_card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        # Player 1 resolves (doesn't counter)
        finished, played_by = self.game_state.play_one_off(1, ace_card, None, 1)
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify all point cards are cleared from both fields
        self.assertEqual(len(self.game_state.fields[0]), 0)
        self.assertEqual(len(self.game_state.fields[1]), 0)
        self.assertEqual(len(self.game_state.discard_pile), 5)  # ACE + 4 point cards

    def test_play_ace_one_off_countered(self):
        """Test playing an Ace as a one-off that gets countered."""
        # Setup initial game state
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        self.hands = [
            [Card("002", Suit.HEARTS, Rank.ACE)],  # Player 0's hand with ACE
            [Card("003", Suit.SPADES, Rank.TWO)],  # Player 1's hand with TWO
        ]
        # Add point cards to both players' fields
        self.fields = [
            [
                Card(
                    "004",
                    Suit.DIAMONDS,
                    Rank.SEVEN,
                    played_by=0,
                    purpose=Purpose.POINTS,
                )
            ],
            [Card("005", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS)],
        ]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Play ACE as one-off
        ace_card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        # Player 1 counters with TWO
        two_card = self.hands[1][0]
        two_card.purpose = Purpose.COUNTER
        two_card.played_by = 1
        finished, played_by = self.game_state.play_one_off(
            1, ace_card, countered_with=two_card
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)

        # Player 0 resolves (accepts counter)
        finished, played_by = self.game_state.play_one_off(0, ace_card, None, 0)
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify point cards remain and ACE + TWO are in discard
        self.assertEqual(len(self.game_state.fields[0]), 1)  # Point card remains
        self.assertEqual(len(self.game_state.fields[1]), 1)  # Point card remains
        self.assertEqual(len(self.game_state.discard_pile), 2)  # ACE + TWO in discard

    def test_counter_one_off(self):
        # Setup initial game state with a one-off card and a counter
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        self.hands = [
            [Card("002", Suit.HEARTS, Rank.ACE)],  # Player 0's hand with ACE
            [Card("003", Suit.SPADES, Rank.TWO)],  # Player 1's hand with TWO
        ]
        self.fields = [
            [
                Card(
                    "004",
                    Suit.DIAMONDS,
                    Rank.SEVEN,
                    played_by=0,
                    purpose=Purpose.POINTS,
                )
            ],
            [Card("005", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS)],
        ]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Player 0 plays ACE as one-off
        ace_card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 0)
        self.game_state.next_player()

        # Player 1 counters with TWO
        two_card = self.hands[1][0]
        two_card.purpose = Purpose.COUNTER
        two_card.played_by = 1  # Set the played_by attribute
        finished, played_by = self.game_state.play_one_off(
            1, ace_card, countered_with=two_card
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)
        self.assertEqual(self.game_state.last_action_played_by, 1)

        # Player 0 resolves (accepts counter)
        finished, played_by = self.game_state.play_one_off(
            0, ace_card, None, last_resolved_by=0
        )
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify ACE was countered and point cards remain
        self.assertEqual(len(self.game_state.fields[0]), 1)
        self.assertEqual(len(self.game_state.fields[1]), 1)
        self.assertEqual(len(self.game_state.discard_pile), 2)  # ACE and TWO in discard

    def test_stacked_counter(self):
        # Setup initial game state with multiple TWOs for stacked countering
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        ace_card = Card("002", Suit.HEARTS, Rank.ACE)
        two_card_1 = Card("003", Suit.DIAMONDS, Rank.TWO)
        two_card_2 = Card("004", Suit.SPADES, Rank.TWO)
        two_card_3 = Card("005", Suit.CLUBS, Rank.TWO)

        self.hands = [
            [ace_card, two_card_2],  # Player 0's hand
            [two_card_1, two_card_3],  # Player 1's hand
        ]
        self.fields = [
            [
                Card(
                    "006",
                    Suit.DIAMONDS,
                    Rank.SEVEN,
                    played_by=0,
                    purpose=Purpose.POINTS,
                )
            ],
            [Card("007", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS)],
        ]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Player 0 plays ACE
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.game_state.next_player()

        # Player 1 counters with first TWO
        two_card_1.purpose = Purpose.COUNTER
        two_card_1.played_by = 1
        finished, played_by = self.game_state.play_one_off(
            1, ace_card, countered_with=two_card_1
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)
        self.game_state.next_player()

        # Player 0 counters the counter with their TWO
        two_card_2.purpose = Purpose.COUNTER
        two_card_2.played_by = 0
        finished, played_by = self.game_state.play_one_off(
            0, two_card_1, countered_with=two_card_2
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 0)
        self.game_state.next_player()

        # Player 1 counters again with their second TWO
        two_card_3.purpose = Purpose.COUNTER
        two_card_3.played_by = 1
        finished, played_by = self.game_state.play_one_off(
            1, two_card_2, countered_with=two_card_3
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)
        self.game_state.next_player()

        # Player 0 resolves (accepts final counter)
        finished, played_by = self.game_state.play_one_off(
            0, two_card_3, None, last_resolved_by=0
        )
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify all cards are in discard and point cards remain
        self.assertEqual(len(self.game_state.fields[0]), 1)
        self.assertEqual(len(self.game_state.fields[1]), 1)
        self.assertEqual(len(self.game_state.discard_pile), 4)  # ACE and three TWOs

    def test_invalid_counter(self):
        # Setup initial game state
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        self.hands = [
            [Card("002", Suit.HEARTS, Rank.ACE)],  # Player 0's hand
            [Card("003", Suit.SPADES, Rank.THREE)],  # Player 1's hand with non-TWO
        ]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Player 0 plays ACE
        ace_card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.game_state.next_player()

        # Player 1 attempts to counter with THREE (should raise exception)
        three_card = self.hands[1][0]
        three_card.purpose = Purpose.COUNTER
        three_card.played_by = 1  # Set the played_by attribute
        with self.assertRaises(Exception) as context:
            self.game_state.play_one_off(1, ace_card, countered_with=three_card)
        self.assertTrue("Counter must be a 2" in str(context.exception))

    def test_play_king_face_card(self):
        """Test playing a King as a face card."""
        # Create a King card
        king = Card("king", Suit.HEARTS, Rank.KING)
        self.hands[0].append(king)

        # Play the King
        self.game_state.play_face_card(king)

        # Verify King is moved to field
        self.assertIn(king, self.game_state.fields[0])
        self.assertNotIn(king, self.game_state.hands[0])
        self.assertEqual(king.purpose, Purpose.FACE_CARD)
        self.assertEqual(king.played_by, 0)

        # Verify target score is reduced
        self.assertEqual(
            self.game_state.get_player_target(0), 14
        )  # 21 -> 14 with one King

    def test_play_multiple_kings(self):
        """Test playing multiple Kings reduces target score correctly."""
        # Create four Kings
        kings = [Card(f"king{i}", Suit.HEARTS, Rank.KING) for i in range(4)]
        self.hands[0].extend(kings)

        # Expected targets after each King
        expected_targets = [14, 10, 5, 0]  # Starting from 21

        # Play Kings one by one
        for i, king in enumerate(kings):
            self.game_state.play_face_card(king)
            self.assertEqual(self.game_state.get_player_target(0), expected_targets[i])

        # Verify all Kings are on the field
        for king in kings:
            self.assertIn(king, self.game_state.fields[0])
            self.assertEqual(king.purpose, Purpose.FACE_CARD)

    def test_play_king_instant_win(self):
        """Test playing a King can lead to instant win if points already meet new target."""
        # Add 11 points to player 0's field
        point_card = Card(
            "points", Suit.HEARTS, Rank.TEN, played_by=0, purpose=Purpose.POINTS
        )
        self.game_state.fields[0].append(point_card)

        # Create and play two Kings to reduce target to 10
        kings = [Card(f"king{i}", Suit.HEARTS, Rank.KING) for i in range(2)]
        self.hands[0].extend(kings)

        # Play first King (target becomes 14, not winning yet)
        self.game_state.play_face_card(kings[0])
        self.assertFalse(self.game_state.is_winner(0))

        # Play second King (target becomes 10, should win with 10 points)
        won = self.game_state.play_face_card(kings[1])
        self.assertTrue(won)
        self.assertTrue(self.game_state.is_winner(0))
        self.assertEqual(self.game_state.winner(), 0)

    def test_play_king_on_opponents_turn(self):
        """Test that Kings can only be played on your own turn."""
        king = Card("king", Suit.HEARTS, Rank.KING)
        self.hands[0].append(king)

        # Set turn to player 1
        self.game_state.turn = 1

        # Try to play King on opponent's turn
        with self.assertRaises(Exception) as context:
            self.game_state.play_face_card(king)

        self.assertIn("Can only play cards from your hand", str(context.exception))

        # Verify game state unchanged
        self.assertIn(king, self.hands[0])
        self.assertNotIn(king, self.game_state.fields[0])
        self.assertEqual(self.game_state.get_player_target(0), 21)
    
    def test_play_queen_face_card(self):
        """Test playing a Queen as a face card."""
        # Set up initial state with face cards on both fields
        hands = [
            [Card("1", Suit.HEARTS, Rank.SIX)],  # Player 0's hand with Six and a point card
            [Card("3", Suit.DIAMONDS, Rank.TWO)],  # Player 1's hand with Two
        ]
        fields = [
            [  # Player 0's field
                Card("4", Suit.HEARTS, Rank.QUEEN),  # Queen face card
                Card("3", Suit.SPADES, Rank.KING),  # King face card
            ],
            [  # Player 1's field
                Card("5", Suit.CLUBS, Rank.JACK),  # Jack face card
                Card("6", Suit.DIAMONDS, Rank.EIGHT),  # Eight face card
            ],
        ]
        deck = []
        discard = []

        # Set up cards on fields
        for card in fields[0]:
            card.purpose = Purpose.FACE_CARD
            card.played_by = 0
        for card in fields[1]:
            card.purpose = Purpose.FACE_CARD
            card.played_by = 1

        game_state = GameState(hands, fields, deck, discard)

        # Play Six as one-off
        six_card = hands[0][0]
        finished, played_by = game_state.play_one_off(0, six_card)
        self.assertFalse(finished)  # Not finished until resolved
        self.assertIsNone(played_by)

        # player 1 tries to counter with a two but can't because of the queen on their field
        two_card = hands[1][0]
        two_card.purpose = Purpose.COUNTER
        two_card.played_by = 1
        with self.assertRaises(Exception) as context:
            game_state.play_one_off(1, six_card, countered_with=two_card)
        self.assertTrue("Cannot counter with a two" in str(context.exception))


    def test_play_six_one_off(self):
        """Test playing a Six as a one-off to destroy all face cards."""
        # Set up initial state with face cards on both fields
        hands = [
            [Card("1", Suit.HEARTS, Rank.SIX)],  # Player 0's hand with Six
            [Card("2", Suit.DIAMONDS, Rank.TWO)],  # Player 1's hand with Two
        ]
        fields = [
            [  # Player 0's field
                Card("3", Suit.SPADES, Rank.KING),  # King face card
                Card("4", Suit.HEARTS, Rank.QUEEN),  # Queen face card
            ],
            [  # Player 1's field
                Card("5", Suit.CLUBS, Rank.JACK),  # Jack face card
                Card("6", Suit.DIAMONDS, Rank.EIGHT),  # Eight face card
            ],
        ]
        deck = []
        discard = []

        # Set up cards on fields
        for card in fields[0]:
            card.purpose = Purpose.FACE_CARD
            card.played_by = 0
        for card in fields[1]:
            card.purpose = Purpose.FACE_CARD
            card.played_by = 1

        game_state = GameState(hands, fields, deck, discard)

        # Play Six as one-off
        six_card = hands[0][0]
        finished, played_by = game_state.play_one_off(0, six_card)
        self.assertFalse(finished)  # Not finished until resolved
        self.assertIsNone(played_by)

        # Player 1 resolves (doesn't counter)
        finished, played_by = game_state.play_one_off(1, six_card, None, 1)
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify all face cards are removed
        self.assertEqual(
            len(game_state.fields[0]), 0
        )  # Player 0's field should be empty
        self.assertEqual(
            len(game_state.fields[1]), 0
        )  # Player 1's field should be empty
        self.assertEqual(
            len(game_state.discard_pile), 5
        )  # Six + 4 face cards in discard

    def test_play_six_one_off_countered(self):
        """Test playing a Six as a one-off that gets countered."""
        # Set up initial state with face cards and a Two to counter
        hands = [
            [Card("1", Suit.HEARTS, Rank.SIX)],  # Player 0's hand with Six
            [Card("2", Suit.DIAMONDS, Rank.TWO)],  # Player 1's hand with Two
        ]
        fields = [
            [Card("3", Suit.SPADES, Rank.KING)],  # Player 0's King
            [Card("4", Suit.HEARTS, Rank.QUEEN)],  # Player 1's Queen
        ]
        deck = []
        discard = []

        # Set up face cards
        fields[0][0].purpose = Purpose.FACE_CARD
        fields[0][0].played_by = 0
        fields[1][0].purpose = Purpose.FACE_CARD
        fields[1][0].played_by = 1

        game_state = GameState(hands, fields, deck, discard)

        # Play Six as one-off
        six_card = hands[0][0]
        finished, played_by = game_state.play_one_off(0, six_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        # Player 1 counters with Two
        two_card = hands[1][0]
        two_card.purpose = Purpose.COUNTER
        two_card.played_by = 1
        finished, played_by = game_state.play_one_off(
            1, six_card, countered_with=two_card
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)

        # Player 0 resolves (accepts counter)
        finished, played_by = game_state.play_one_off(0, six_card, None, 0)
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify face cards remain and Six + Two are in discard
        self.assertEqual(len(game_state.fields[0]), 1)  # King remains
        self.assertEqual(len(game_state.fields[1]), 1)  # Queen remains
        self.assertEqual(len(game_state.discard_pile), 2)  # Six + Two in discard

    def test_play_ace_one_off(self):
        """Test playing an Ace as a one-off to destroy all point cards."""
        # Setup initial game state with point cards on both players' fields
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        self.hands = [
            [Card("002", Suit.HEARTS, Rank.ACE)],  # Player 0's hand with ACE
            [Card("003", Suit.SPADES, Rank.TWO)],  # Player 1's hand
        ]
        # Add point cards to both players' fields
        self.fields = [
            [
                Card(
                    "004",
                    Suit.DIAMONDS,
                    Rank.SEVEN,
                    played_by=0,
                    purpose=Purpose.POINTS,
                ),
                Card("005", Suit.CLUBS, Rank.FOUR, played_by=0, purpose=Purpose.POINTS),
            ],
            [
                Card(
                    "006", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS
                ),
                Card(
                    "007", Suit.SPADES, Rank.THREE, played_by=1, purpose=Purpose.POINTS
                ),
            ],
        ]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Play ACE as one-off
        ace_card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        # Player 1 resolves (doesn't counter)
        finished, played_by = self.game_state.play_one_off(1, ace_card, None, 1)
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify all point cards are cleared from both fields
        self.assertEqual(len(self.game_state.fields[0]), 0)
        self.assertEqual(len(self.game_state.fields[1]), 0)
        self.assertEqual(len(self.game_state.discard_pile), 5)  # ACE + 4 point cards

    def test_play_ace_one_off_countered(self):
        """Test playing an Ace as a one-off that gets countered."""
        # Setup initial game state
        self.deck = [Card("001", Suit.CLUBS, Rank.THREE)]
        self.hands = [
            [Card("002", Suit.HEARTS, Rank.ACE)],  # Player 0's hand with ACE
            [Card("003", Suit.SPADES, Rank.TWO)],  # Player 1's hand with TWO
        ]
        # Add point cards to both players' fields
        self.fields = [
            [
                Card(
                    "004",
                    Suit.DIAMONDS,
                    Rank.SEVEN,
                    played_by=0,
                    purpose=Purpose.POINTS,
                )
            ],
            [Card("005", Suit.HEARTS, Rank.NINE, played_by=1, purpose=Purpose.POINTS)],
        ]
        self.discard_pile = []

        self.game_state = GameState(
            self.hands, self.fields, self.deck, self.discard_pile
        )

        # Play ACE as one-off
        ace_card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, ace_card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        # Player 1 counters with TWO
        two_card = self.hands[1][0]
        two_card.purpose = Purpose.COUNTER
        two_card.played_by = 1
        finished, played_by = self.game_state.play_one_off(
            1, ace_card, countered_with=two_card
        )
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)

        # Player 0 resolves (accepts counter)
        finished, played_by = self.game_state.play_one_off(0, ace_card, None, 0)
        self.assertTrue(finished)
        self.assertIsNone(played_by)

        # Verify point cards remain and ACE + TWO are in discard
        self.assertEqual(len(self.game_state.fields[0]), 1)  # Point card remains
        self.assertEqual(len(self.game_state.fields[1]), 1)  # Point card remains
        self.assertEqual(len(self.game_state.discard_pile), 2)  # ACE + TWO in discard

    @patch("builtins.input")
    def test_play_three_one_off(self, mock_input):
        """Test playing a Three as a one-off to take a card from discard pile."""
        # Setup initial state with cards in discard pile
        three_card = Card("1", Suit.HEARTS, Rank.THREE)
        two_card = Card("2", Suit.DIAMONDS, Rank.TWO)
        king_card = Card("3", Suit.SPADES, Rank.KING)
        queen_card = Card("4", Suit.HEARTS, Rank.QUEEN)
        jack_card = Card("5", Suit.CLUBS, Rank.JACK)

        hands = [
            [three_card],  # Player 0's hand with Three
            [two_card],  # Player 1's hand with Two
        ]
        fields = [[], []]
        deck = []
        discard = [
            king_card,  # King in discard
            queen_card,  # Queen in discard
            jack_card,  # Jack in discard
        ]

        game_state = GameState(hands, fields, deck, discard)

        # Mock input to select first card from discard
        mock_input.return_value = "0"

        # Play Three as one-off
        turn_finished, played_by = game_state.play_one_off(
            0, three_card, None, 1
        )  # Player 1 resolves

        # Verify Three is moved to discard
        self.assertIn(three_card, game_state.discard_pile)
        self.assertNotIn(three_card, game_state.hands[0])

        # Verify King was taken from discard
        self.assertIn(king_card, game_state.hands[0])
        self.assertNotIn(king_card, game_state.discard_pile)

        # Verify other cards remain in discard
        self.assertEqual(len(game_state.discard_pile), 3)  # Three + Queen + Jack
        self.assertIn(queen_card, game_state.discard_pile)
        self.assertIn(jack_card, game_state.discard_pile)

    @patch("builtins.input")
    def test_play_three_one_off_empty_discard(self, mock_input):
        """Test playing a Three as a one-off with empty discard pile."""
        # Setup initial state with empty discard pile
        three_card = Card("1", Suit.HEARTS, Rank.THREE)
        two_card = Card("2", Suit.DIAMONDS, Rank.TWO)

        hands = [
            [three_card],  # Player 0's hand with Three
            [two_card],  # Player 1's hand with Two
        ]
        fields = [[], []]
        deck = []
        discard = []

        game_state = GameState(hands, fields, deck, discard)

        # Play Three as one-off
        turn_finished, played_by = game_state.play_one_off(
            0, three_card, None, 1
        )  # Player 1 resolves

        # Verify Three is moved to discard
        self.assertIn(three_card, game_state.discard_pile)
        self.assertNotIn(three_card, game_state.hands[0])

        # Verify hand remains unchanged (no card taken as discard was empty)
        self.assertEqual(len(game_state.hands[0]), 0)

    @patch("builtins.input")
    def test_play_three_one_off_with_counter(self, mock_input):
        """Test playing a Three as a one-off that gets countered."""
        # Setup initial state
        three_card = Card("1", Suit.HEARTS, Rank.THREE)
        two_card = Card("2", Suit.DIAMONDS, Rank.TWO)
        king_card = Card("3", Suit.SPADES, Rank.KING)

        hands = [
            [three_card],  # Player 0's hand with Three
            [two_card],  # Player 1's hand with Two
        ]
        fields = [[], []]
        deck = []
        discard = [king_card]  # King in discard

        game_state = GameState(hands, fields, deck, discard)

        # Play Three as one-off
        turn_finished, played_by = game_state.play_one_off(0, three_card)
        self.assertFalse(turn_finished)
        self.assertIsNone(played_by)
        game_state.next_player()

        # Counter with Two
        two_card.purpose = Purpose.COUNTER
        two_card.played_by = 1
        turn_finished, played_by = game_state.play_one_off(
            1, three_card, countered_with=two_card
        )
        self.assertFalse(turn_finished)
        self.assertEqual(played_by, 1)
        game_state.next_player()

        # Original player accepts counter
        turn_finished, played_by = game_state.play_one_off(
            0, two_card, None, last_resolved_by=0
        )
        self.assertTrue(turn_finished)
        self.assertIsNone(played_by)

        # Verify both cards are in discard
        self.assertIn(three_card, game_state.discard_pile)
        self.assertIn(two_card, game_state.discard_pile)

        # Verify King remains in discard (effect not applied)
        self.assertIn(king_card, game_state.discard_pile)
        self.assertEqual(len(game_state.discard_pile), 3)

    @patch("builtins.input")
    def test_play_three_one_off_invalid_selection(self, mock_input):
        """Test playing a Three as a one-off with invalid card selection."""
        # Setup initial state
        three_card = Card("1", Suit.HEARTS, Rank.THREE)
        two_card = Card("2", Suit.DIAMONDS, Rank.TWO)
        king_card = Card("3", Suit.SPADES, Rank.KING)
        queen_card = Card("4", Suit.HEARTS, Rank.QUEEN)

        hands = [
            [three_card],  # Player 0's hand with Three
            [two_card],  # Player 1's hand with Two
        ]
        fields = [[], []]
        deck = []
        discard = [
            king_card,
            queen_card,
        ]

        game_state = GameState(hands, fields, deck, discard)

        # Mock invalid input followed by valid input
        mock_input.side_effect = ["invalid", "-1", "999", "0"]

        # Play Three as one-off
        turn_finished, played_by = game_state.play_one_off(
            0, three_card, None, 1
        )  # Player 1 resolves

        # Verify Three is moved to discard
        self.assertIn(three_card, game_state.discard_pile)
        self.assertNotIn(three_card, game_state.hands[0])

        # Verify King was taken from discard (after invalid attempts)
        self.assertIn(king_card, game_state.hands[0])
        self.assertNotIn(king_card, game_state.discard_pile)

        # Verify Queen remains in discard
        self.assertIn(queen_card, game_state.discard_pile)
        self.assertEqual(len(game_state.discard_pile), 2)  # Three + Queen


if __name__ == "__main__":
    unittest.main()
