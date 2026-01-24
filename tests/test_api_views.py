from game.action import Action, ActionType
from game.card import Card, Purpose, Rank, Suit
from game.game_state import GameState
from server.views import action_view, game_state_view


def _card(card_id: str, suit: Suit, rank: Rank) -> Card:
    return Card(id=card_id, suit=suit, rank=rank)


def test_action_view_serialization() -> None:
    card = _card("c1", Suit.SPADES, Rank.TEN)
    action = Action(action_type=ActionType.POINTS, played_by=0, card=card)

    view = action_view(action, action_id=3).model_dump()

    assert view["id"] == 3
    assert view["type"] == ActionType.POINTS.value
    assert view["played_by"] == 0
    assert view["card"]["id"] == "c1"


def test_game_state_view_serialization_hides_hand() -> None:
    hand0 = [_card("h0", Suit.HEARTS, Rank.ACE)]
    hand1 = [_card("h1", Suit.CLUBS, Rank.FIVE), _card("h2", Suit.SPADES, Rank.TWO)]

    field0 = [_card("f0", Suit.CLUBS, Rank.SEVEN)]
    field0[0].purpose = Purpose.POINTS
    field0[0].played_by = 0

    field1 = [_card("f1", Suit.DIAMONDS, Rank.QUEEN)]
    field1[0].purpose = Purpose.FACE_CARD
    field1[0].played_by = 1

    deck = [_card("d0", Suit.SPADES, Rank.THREE)]
    discard = [_card("x0", Suit.HEARTS, Rank.NINE)]

    state = GameState(
        hands=[hand0, hand1],
        fields=[field0, field1],
        deck=deck,
        discard_pile=discard,
    )

    view = game_state_view(state, hide_player_hand=1).model_dump()

    assert view["deck_count"] == 1
    assert view["discard_count"] == 1
    assert view["hand_counts"] == [1, 2]
    assert view["hands"][1] == []
    assert view["scores"][0] == 7
    assert view["targets"][0] == 21
