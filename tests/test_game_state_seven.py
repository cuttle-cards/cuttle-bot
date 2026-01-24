import unittest
from typing import List

from game.action import Action, ActionType
from game.card import Card, Purpose, Rank, Suit
from game.game_history import GameHistory
from game.game_state import GameState


class TestGameStateSeven(unittest.TestCase):
    def _resolve_seven_one_off(self, state: GameState, chosen_card: Card) -> None:
        state.apply_one_off_effect(Card("seven", Suit.SPADES, Rank.SEVEN))
        action = next(
            act
            for act in state.get_legal_actions()
            if act.action_type == ActionType.ONE_OFF and act.card == chosen_card
        )
        turn_finished, should_stop, _winner = state.update_state(action)
        self.assertFalse(turn_finished)
        self.assertFalse(should_stop)
        self.assertTrue(state.resolving_one_off)

        state.next_player()
        resolve_action = next(
            act for act in state.get_legal_actions() if act.action_type == ActionType.RESOLVE
        )
        state.update_state(resolve_action)
    def test_seven_state_serialization_round_trip(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        fields: List[List[Card]] = [[], []]
        deck: List[Card] = [Card("2", Suit.CLUBS, Rank.ACE)]
        discard: List[Card] = []

        state = GameState(hands, fields, deck, discard)
        state.resolving_seven = True
        state.pending_seven_player = 0
        state.pending_seven_cards = [Card("3", Suit.SPADES, Rank.NINE)]
        state.pending_seven_requires_discard = True

        payload = state.to_dict()
        restored = GameState.from_dict(payload)

        self.assertTrue(restored.resolving_seven)
        self.assertEqual(restored.pending_seven_player, 0)
        self.assertEqual(len(restored.pending_seven_cards), 1)
        self.assertEqual(restored.pending_seven_cards[0].rank, Rank.NINE)
        self.assertTrue(restored.pending_seven_requires_discard)

    def test_discard_revealed_action_description(self) -> None:
        history = GameHistory()
        card = Card("1", Suit.HEARTS, Rank.SEVEN)
        history.record_action(
            player=0,
            action_type=ActionType.DISCARD_REVEALED,
            card=card,
            source="deck",
            destination="discard_pile",
        )

        self.assertIn("discards revealed", history.entries[-1].description)
        action = Action(ActionType.DISCARD_REVEALED, played_by=0, card=card)
        self.assertIn("Discard revealed", str(action))

    def test_seven_reveal_sets_pending_state(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        fields: List[List[Card]] = [[], []]
        deck: List[Card] = [
            Card("2", Suit.CLUBS, Rank.TWO),
            Card("3", Suit.SPADES, Rank.NINE),
        ]
        state = GameState(hands, fields, deck, [])
        state.apply_one_off_effect(hands[0][0])

        self.assertTrue(state.resolving_seven)
        self.assertEqual(state.pending_seven_player, 0)
        self.assertEqual(len(state.pending_seven_cards), 2)
        self.assertEqual(state.pending_seven_cards[0].rank, Rank.NINE)
        self.assertFalse(state.pending_seven_requires_discard)

    def test_seven_requires_discard_when_unplayable(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        fields: List[List[Card]] = [[], []]
        deck: List[Card] = [
            Card("2", Suit.CLUBS, Rank.JACK),
            Card("3", Suit.SPADES, Rank.JACK),
        ]
        state = GameState(hands, fields, deck, [])
        state.apply_one_off_effect(hands[0][0])

        self.assertTrue(state.resolving_seven)
        self.assertTrue(state.pending_seven_requires_discard)
        actions = state.get_legal_actions()
        self.assertTrue(all(action.action_type == ActionType.DISCARD_REVEALED for action in actions))

    def test_seven_single_unplayable_auto_discards(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        fields: List[List[Card]] = [[], []]
        deck: List[Card] = [Card("2", Suit.CLUBS, Rank.JACK)]
        discard: List[Card] = []
        state = GameState(hands, fields, deck, discard)

        state.apply_one_off_effect(hands[0][0])

        self.assertFalse(state.resolving_seven)
        self.assertEqual(len(state.deck), 0)
        self.assertEqual(len(state.discard_pile), 1)
        self.assertEqual(state.discard_pile[0].rank, Rank.JACK)

    def test_seven_choose_points_action(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        fields: List[List[Card]] = [[], []]
        top_card = Card("2", Suit.HEARTS, Rank.NINE)
        second_card = Card("3", Suit.CLUBS, Rank.THREE)
        deck: List[Card] = [second_card, top_card]
        state = GameState(hands, fields, deck, [])

        state.apply_one_off_effect(hands[0][0])
        action = next(
            act
            for act in state.get_legal_actions()
            if act.action_type == ActionType.POINTS and act.card == top_card
        )
        turn_finished, should_stop, _winner = state.update_state(action)

        self.assertTrue(turn_finished)
        self.assertFalse(should_stop)
        self.assertFalse(state.resolving_seven)
        self.assertIn(top_card, state.fields[0])
        self.assertEqual(state.deck[-1], second_card)

    def test_seven_choose_scuttle_action(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        target = Card("t", Suit.DIAMONDS, Rank.FIVE, played_by=1, purpose=Purpose.POINTS)
        fields: List[List[Card]] = [[], [target]]
        top_card = Card("2", Suit.SPADES, Rank.SEVEN)
        deck: List[Card] = [top_card]
        state = GameState(hands, fields, deck, [])

        state.apply_one_off_effect(hands[0][0])
        action = next(
            act for act in state.get_legal_actions() if act.action_type == ActionType.SCUTTLE
        )
        state.update_state(action)

        self.assertFalse(state.resolving_seven)
        self.assertIn(target, state.discard_pile)
        self.assertIn(top_card, state.discard_pile)
        self.assertNotIn(target, state.fields[1])

    def test_seven_choose_face_card_action(self) -> None:
        hands: List[List[Card]] = [[Card("1", Suit.HEARTS, Rank.SEVEN)], []]
        top_card = Card("2", Suit.HEARTS, Rank.KING)
        deck: List[Card] = [top_card]
        state = GameState(hands, [[], []], deck, [])

        state.apply_one_off_effect(hands[0][0])
        action = next(
            act for act in state.get_legal_actions() if act.action_type == ActionType.FACE_CARD
        )
        state.update_state(action)

        self.assertIn(top_card, state.fields[0])
        self.assertFalse(state.resolving_seven)

    def test_seven_selects_ace_one_off(self) -> None:
        ace = Card("ace", Suit.HEARTS, Rank.ACE)
        deck: List[Card] = [
            Card("f1", Suit.CLUBS, Rank.TWO),
            Card("f2", Suit.SPADES, Rank.THREE),
            ace,
        ]
        fields: List[List[Card]] = [
            [Card("p1", Suit.HEARTS, Rank.TEN, played_by=0, purpose=Purpose.POINTS)],
            [Card("p2", Suit.CLUBS, Rank.NINE, played_by=1, purpose=Purpose.POINTS)],
        ]
        state = GameState([[ ], []], fields, deck, [], input_mode="api")

        self._resolve_seven_one_off(state, ace)

        self.assertEqual(len(state.fields[0]), 0)
        self.assertEqual(len(state.fields[1]), 0)
        self.assertIn(ace, state.discard_pile)

    def test_seven_selects_three_one_off(self) -> None:
        three = Card("three", Suit.CLUBS, Rank.THREE)
        deck: List[Card] = [
            Card("f1", Suit.HEARTS, Rank.TWO),
            Card("f2", Suit.SPADES, Rank.FOUR),
            three,
        ]
        discard: List[Card] = [Card("d1", Suit.HEARTS, Rank.FIVE)]
        state = GameState([[ ], []], [[], []], deck, discard, input_mode="api")

        self._resolve_seven_one_off(state, three)

        self.assertTrue(state.resolving_three)
        self.assertEqual(state.pending_three_player, 0)
        self.assertIn(three, state.discard_pile)

    def test_seven_selects_four_one_off(self) -> None:
        four = Card("four", Suit.DIAMONDS, Rank.FOUR)
        deck: List[Card] = [
            Card("f1", Suit.HEARTS, Rank.TWO),
            Card("f2", Suit.SPADES, Rank.FIVE),
            four,
        ]
        hands: List[List[Card]] = [
            [],
            [Card("h1", Suit.CLUBS, Rank.NINE), Card("h2", Suit.SPADES, Rank.EIGHT)],
        ]
        state = GameState(hands, [[], []], deck, [], input_mode="api")

        self._resolve_seven_one_off(state, four)

        self.assertTrue(state.resolving_four)
        self.assertEqual(state.pending_four_player, 1)
        self.assertEqual(state.pending_four_count, 2)
        self.assertIn(four, state.discard_pile)

    def test_seven_selects_five_one_off(self) -> None:
        five = Card("five", Suit.HEARTS, Rank.FIVE)
        deck: List[Card] = [
            Card("f1", Suit.CLUBS, Rank.ACE),
            Card("f2", Suit.SPADES, Rank.TWO),
            five,
        ]
        state = GameState([[], []], [[], []], deck, [], input_mode="api")

        self._resolve_seven_one_off(state, five)

        self.assertEqual(len(state.hands[0]), 2)
        self.assertIn(five, state.discard_pile)

    def test_seven_selects_six_one_off(self) -> None:
        six = Card("six", Suit.HEARTS, Rank.SIX)
        deck: List[Card] = [
            Card("f1", Suit.CLUBS, Rank.ACE),
            Card("f2", Suit.SPADES, Rank.TWO),
            six,
        ]
        fields: List[List[Card]] = [
            [Card("k1", Suit.HEARTS, Rank.KING, played_by=0, purpose=Purpose.FACE_CARD)],
            [Card("q1", Suit.CLUBS, Rank.QUEEN, played_by=1, purpose=Purpose.FACE_CARD)],
        ]
        state = GameState([[], []], fields, deck, [], input_mode="api")

        self._resolve_seven_one_off(state, six)

        self.assertEqual(len(state.fields[0]), 0)
        self.assertEqual(len(state.fields[1]), 0)
        self.assertIn(six, state.discard_pile)

    def test_seven_selects_seven_one_off(self) -> None:
        seven = Card("seven2", Suit.HEARTS, Rank.SEVEN)
        deck: List[Card] = [
            Card("f1", Suit.CLUBS, Rank.ACE),
            Card("f2", Suit.SPADES, Rank.TWO),
            seven,
        ]
        state = GameState([[], []], [[], []], deck, [], input_mode="api")

        self._resolve_seven_one_off(state, seven)

        self.assertTrue(state.resolving_seven)
        self.assertGreaterEqual(len(state.pending_seven_cards), 1)
        self.assertIn(seven, state.discard_pile)


if __name__ == "__main__":
    unittest.main()
