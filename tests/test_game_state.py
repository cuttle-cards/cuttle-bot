import unittest
from game.card import Card, Suit, Rank
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

    def test_is_winner(self):
        self.assertFalse(self.game_state.is_winner(0))
        self.assertFalse(self.game_state.is_winner(1))

    def test_winner(self):
        self.assertIsNone(self.game_state.winner())

    def test_is_stalemate(self):
        self.assertFalse(self.game_state.is_stalemate())

    def test_draw_card(self):
        self.game_state.draw_card()
        self.assertEqual(len(self.game_state.hands[0]), 6)
        self.assertEqual(len(self.game_state.deck), 9)

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

if __name__ == '__main__':
    unittest.main()
