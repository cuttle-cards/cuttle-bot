import unittest
from game.game_state import GameState
from game.card import Card, Suit, Rank, Purpose
from game.action import Action, ActionType


class TestGameStateScuttle(unittest.TestCase):
    def setUp(self):
        """Set up a game state for testing scuttling."""
        # Create hands for both players
        self.p0_hand = [
            Card("1", Suit.HEARTS, Rank.TEN),  # Point card
            Card("2", Suit.SPADES, Rank.NINE),  # Point card
            Card("3", Suit.DIAMONDS, Rank.JACK),  # Face card
            Card("4", Suit.CLUBS, Rank.QUEEN),  # Face card
            Card("5", Suit.HEARTS, Rank.KING),  # Face card
        ]
        self.p1_hand = [
            Card("6", Suit.DIAMONDS, Rank.EIGHT),  # Point card
            Card("7", Suit.CLUBS, Rank.SEVEN),  # Point card
        ]

        # Create fields with some point cards
        self.p0_field = []
        self.p1_field = [
            Card(
                "8",
                suit=Suit.HEARTS,
                rank=Rank.SEVEN,
                purpose=Purpose.POINTS,
                played_by=1,
            ),  # 7 points
            Card(
                "9",
                suit=Suit.DIAMONDS,
                rank=Rank.FIVE,
                purpose=Purpose.POINTS,
                played_by=1,
            ),  # 5 points
        ]

        # Initialize game state
        self.game_state = GameState(
            hands=[self.p0_hand, self.p1_hand],
            fields=[self.p0_field, self.p1_field],
            deck=[],  # Empty deck for simplicity
            discard_pile=[],  # Empty discard pile
        )
        self.game_state.turn = 0  # P0's turn

    def test_scuttle_with_point_card(self):
        """Test that point cards (2-10) can scuttle."""
        actions = self.game_state.get_legal_actions()
        print(actions)
        scuttle_actions = [a for a in actions if a.action_type == ActionType.SCUTTLE]
        print(scuttle_actions)

        # Ten of Hearts should be able to scuttle Seven of Hearts
        self.assertTrue(
            any(
                a.card.rank == Rank.TEN and a.target.rank == Rank.SEVEN
                for a in scuttle_actions
            ),
            "Ten of Hearts should be able to scuttle Seven of Hearts",
        )

        # Nine of Spades should be able to scuttle Seven of Hearts
        self.assertTrue(
            any(
                a.card.rank == Rank.NINE and a.target.rank == Rank.SEVEN
                for a in scuttle_actions
            ),
            "Nine of Spades should be able to scuttle Seven of Hearts",
        )

    def test_scuttle_with_face_cards_not_allowed(self):
        """Test that face cards (Jack, Queen, King) cannot scuttle."""
        actions = self.game_state.get_legal_actions()
        scuttle_actions = [a for a in actions if a.action_type == ActionType.SCUTTLE]

        # No face cards should be in scuttle actions
        self.assertFalse(
            any(
                a.card.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]
                for a in scuttle_actions
            ),
            "Face cards should not be allowed to scuttle",
        )

    def test_scuttle_with_equal_value_higher_suit(self):
        """Test scuttling with equal value but higher suit."""
        # Add a Seven of Spades to P0's hand (Spades > Hearts)
        seven_spades = Card("10", Suit.SPADES, Rank.SEVEN)
        self.game_state.hands[0].append(seven_spades)

        actions = self.game_state.get_legal_actions()
        scuttle_actions = [a for a in actions if a.action_type == ActionType.SCUTTLE]

        # Seven of Diamonds should be able to scuttle Seven of Hearts
        self.assertTrue(
            any(
                a.card == seven_spades
                and a.target.rank == Rank.SEVEN
                and a.target.suit == Suit.HEARTS
                for a in scuttle_actions
            ),
            "Seven of Spades should be able to scuttle Seven of Hearts (higher suit)",
        )

    def test_scuttle_with_equal_value_lower_suit_not_allowed(self):
        """Test that equal value with lower suit cannot scuttle."""
        # Add a Seven of Clubs to P0's hand (Clubs < Hearts)
        seven_clubs = Card("11", Suit.CLUBS, Rank.SEVEN)
        self.game_state.hands[0].append(seven_clubs)

        actions = self.game_state.get_legal_actions()
        scuttle_actions = [a for a in actions if a.action_type == ActionType.SCUTTLE]

        # Seven of Clubs should not be able to scuttle Seven of Hearts
        self.assertFalse(
            any(
                a.card == seven_clubs
                and a.target.rank == Rank.SEVEN
                and a.target.suit == Suit.HEARTS
                for a in scuttle_actions
            ),
            "Seven of Clubs should not be able to scuttle Seven of Hearts (lower suit)",
        )

    def test_scuttle_with_lower_value_not_allowed(self):
        """Test that lower value cards cannot scuttle."""
        # Add a Four of Hearts to P0's hand
        four_hearts = Card("12", Suit.HEARTS, Rank.FOUR)
        self.game_state.hands[0].append(four_hearts)

        actions = self.game_state.get_legal_actions()
        scuttle_actions = [a for a in actions if a.action_type == ActionType.SCUTTLE]

        # Four should not be able to scuttle Five or Seven
        self.assertFalse(
            any(a.card == four_hearts for a in scuttle_actions),
            "Lower value cards should not be able to scuttle higher value cards",
        )

    def test_scuttle_only_point_cards(self):
        """Test that only point cards on the field can be scuttled."""
        # Add a face card to opponent's field
        queen_hearts = Card(
            id="13", suit=Suit.HEARTS, rank=Rank.QUEEN, purpose=Purpose.FACE_CARD, played_by=1
        )
        self.game_state.fields[1].append(queen_hearts)

        actions = self.game_state.get_legal_actions()
        scuttle_actions = [a for a in actions if a.action_type == ActionType.SCUTTLE]

        # No actions should target the Queen
        self.assertFalse(
            any(a.target == queen_hearts for a in scuttle_actions),
            "Face cards should not be targetable for scuttling",
        )

        # Should still be able to scuttle point cards
        self.assertTrue(
            any(a.target.rank == Rank.SEVEN for a in scuttle_actions),
            "Point cards should still be scuttleable when face cards are present",
        )
