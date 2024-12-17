import unittest
from game.card import Card, Purpose, Suit, Rank
from game.game_state import GameState

class TestGameState(unittest.TestCase):

    def setUp(self):
        # Create a sample deck and hands for testing
        self.deck = [Card(str(i), Suit.CLUBS, Rank.ACE) for i in range(10)]
        self.hands = [[Card(str(i), Suit.HEARTS, Rank.TWO) for i in range(5)],
                      [Card(str(i), Suit.SPADES, Rank.THREE) for i in range(6)]]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(self.hands, self.fields, self.deck, self.discard_pile)

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
        p0_new_cards = [Card("", Suit.HEARTS, Rank.KING, played_by=0), Card("", Suit.CLUBS, Rank.KING, played_by=0), Card("", Suit.DIAMONDS, Rank.TEN, purpose=Purpose.POINTS, played_by=0)]
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
        target = Card("target", Suit.HEARTS, Rank.TWO, played_by=1)
        self.game_state.fields[1].append(target)
        self.game_state.scuttle(card, target)
        self.assertIn(card, self.game_state.discard_pile)
        self.assertIn(target, self.game_state.discard_pile)
        self.assertNotIn(target, self.game_state.fields[1])

    def test_play_one_off(self):
        counter_card = Card("counter", Suit.HEARTS, Rank.TWO, played_by=1, purpose=Purpose.COUNTER)
        
        self.hands[1].append(counter_card)

        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)

        finished, played_by = self.game_state.play_one_off(1, card, countered_with=counter_card)
        self.assertFalse(finished)
        self.assertEqual(played_by, 1)
        self.assertNotEqual(self.game_state.turn, played_by)

        counter_card_0 = Card("counter2", Suit.DIAMONDS, Rank.TWO, played_by=0, purpose=Purpose.COUNTER)
        self.hands[0].append(counter_card_0)

        finished, played_by = self.game_state.play_one_off(0, card, countered_with=counter_card_0)
        self.assertFalse(finished)
        self.assertEqual(played_by, 0)

        finished, played_by = self.game_state.play_one_off(1, card, last_resolved_by=1)
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertIn(card, self.game_state.discard_pile)
        self.assertNotIn(card, self.game_state.hands[0])
    

    def test_play_five_one_off(self):
        self.deck = [Card("001", Suit.CLUBS, Rank.ACE), Card("002", Suit.CLUBS, Rank.TWO)]
        self.hands = [[Card("003", Suit.HEARTS, Rank.FIVE)],
                      [Card("004", Suit.SPADES, Rank.SIX)]]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(self.hands, self.fields, self.deck, self.discard_pile)

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
        finished, played_by = self.game_state.play_one_off(1, card, None, last_resolved_by=self.game_state.current_action_player)
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 1)
        self.assertEqual(len(self.game_state.hands[0]), 2)

        self.game_state.next_turn()
        self.assertEqual(self.game_state.turn, 1)


    def test_play_five_one_off_with_eight_cards(self):
        self.deck = [Card("001", Suit.CLUBS, Rank.ACE), Card("002", Suit.CLUBS, Rank.TWO)]
        self.hands = [[Card("003", Suit.HEARTS, Rank.FIVE), Card("004", Suit.HEARTS, Rank.ACE), Card("005", Suit.HEARTS, Rank.TWO), Card("006", Suit.HEARTS, Rank.THREE), Card("007", Suit.HEARTS, Rank.FOUR), Card("008", Suit.HEARTS, Rank.SIX), Card("009", Suit.HEARTS, Rank.SEVEN)], []]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(self.hands, self.fields, self.deck, self.discard_pile)

        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 0)
        self.assertEqual(self.game_state.current_action_player, 0)
        self.game_state.next_player()
        self.assertEqual(self.game_state.current_action_player, 1)

        finished, played_by = self.game_state.play_one_off(1, card, None, last_resolved_by=self.game_state.current_action_player)
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 1)

        self.assertEqual(len(self.game_state.hands[0]), 8)
    

        self.deck = [Card("001", Suit.CLUBS, Rank.ACE), Card("002", Suit.CLUBS, Rank.TWO)]
        self.hands = [[Card("003", Suit.HEARTS, Rank.FIVE), Card("004", Suit.HEARTS, Rank.ACE), Card("005", Suit.HEARTS, Rank.TWO), Card("006", Suit.HEARTS, Rank.THREE), Card("007", Suit.HEARTS, Rank.FOUR), Card("008", Suit.HEARTS, Rank.SIX), Card("009", Suit.HEARTS, Rank.SEVEN), Card("010", Suit.HEARTS, Rank.EIGHT)], []]
        self.fields = [[], []]
        self.discard_pile = []

        self.game_state = GameState(self.hands, self.fields, self.deck, self.discard_pile)

        card = self.hands[0][0]
        finished, played_by = self.game_state.play_one_off(0, card)
        self.assertFalse(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 0)
        self.assertEqual(self.game_state.current_action_player, 0)
        self.game_state.next_player()
        self.assertEqual(self.game_state.current_action_player, 1)

        finished, played_by = self.game_state.play_one_off(1, card, None, last_resolved_by=self.game_state.current_action_player)
        self.assertTrue(finished)
        self.assertIsNone(played_by)
        self.assertEqual(self.game_state.turn, 0)
        self.assertEqual(self.game_state.last_action_played_by, 1)

        self.assertEqual(len(self.game_state.hands[0]), 8)

if __name__ == '__main__':
    unittest.main()
