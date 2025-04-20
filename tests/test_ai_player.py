import asyncio
import unittest
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from game.action import Action, ActionType
from game.ai_player import AIPlayer
from game.card import Card, Rank, Suit
from game.game_state import GameState


class TestAIPlayer(unittest.IsolatedAsyncioTestCase):
    ai_player: AIPlayer
    p0_cards: List[Card]
    p1_cards: List[Card]
    deck: List[Card]
    game_state: GameState

    def setUp(self) -> None:
        self.ai_player = AIPlayer(retry_delay=0.1, max_retries=1)
        # Create a simple game state for testing
        self.p0_cards = [
            Card("1", Suit.HEARTS, Rank.KING),
            Card("2", Suit.SPADES, Rank.TEN),
        ]
        self.p1_cards = [
            Card("3", Suit.DIAMONDS, Rank.QUEEN),
            Card("4", Suit.CLUBS, Rank.FIVE),
        ]
        self.deck = [
            Card("5", Suit.HEARTS, Rank.TWO),
            Card("6", Suit.SPADES, Rank.THREE),
        ]
        self.game_state = GameState(
            hands=[self.p0_cards, self.p1_cards],
            fields=[[], []],
            deck=self.deck,
            discard_pile=[],
        )

    @pytest.mark.timeout(15)
    def test_format_game_state(self) -> None:
        """Test that game state is formatted correctly for the LLM."""
        legal_actions: List[Action] = [
            Action(action_type=ActionType.DRAW, card=None, target=None, played_by=1),
            Action(action_type=ActionType.POINTS, card=self.p1_cards[1], target=None, played_by=1),
        ]

        formatted_state = self.ai_player._format_game_state(
            self.game_state, legal_actions
        )

        # Check that key information is included in the formatted state
        self.assertIn("AI Hand:", formatted_state)
        self.assertIn("Queen of Diamonds", formatted_state)
        self.assertIn("Five of Clubs", formatted_state)
        self.assertIn("Legal Actions:", formatted_state)
        self.assertIn("Draw a card from deck", formatted_state)
        self.assertIn("Play Five of Clubs as points", formatted_state)

    @pytest.mark.timeout(10)
    @patch("ollama.chat")
    @patch("game.ai_player.AIPlayer._format_game_state")
    async def test_get_action_success(self, mock_format_game_state: Mock, mock_chat: Mock) -> None:
        """Test successful action selection by AI."""
        legal_actions: List[Action] = [
            Action(action_type=ActionType.DRAW, card=None, target=None, played_by=1),
            Action(action_type=ActionType.POINTS, card=self.p1_cards[1], target=None, played_by=1),
        ]

        mock_chat.return_value = {
            'message': {
                'content': "I choose to play Five of Clubs as points to start building my score. Action number: 1",
                'role': 'assistant'
            }
        }

        mock_format_game_state.return_value = "mock game state"

        # Get AI action
        action = await self.ai_player.get_action(self.game_state, legal_actions)

        # Verify the correct action was chosen
        self.assertEqual(action, legal_actions[1])
        self.assertEqual(action.action_type, ActionType.POINTS)
        self.assertEqual(action.card, self.p1_cards[1])

    @pytest.mark.timeout(10)
    @patch("ollama.chat")
    @patch("game.ai_player.AIPlayer._format_game_state")
    async def test_get_action_invalid_response(self, mock_format_game_state: Mock, mock_chat: Mock) -> None:
        """Test handling of invalid LLM response."""
        legal_actions: List[Action] = [
            Action(action_type=ActionType.DRAW, card=None, target=None, played_by=1),
            Action(action_type=ActionType.POINTS, card=self.p1_cards[1], target=None, played_by=1),
        ]

        mock_chat.return_value = {
            'message': {
                'content': "I am not sure what to do.",
                'role': 'assistant'
            }
        }

        mock_format_game_state.return_value = "mock game state"

        # Get AI action - should default to first legal action
        action = await self.ai_player.get_action(self.game_state, legal_actions)

        # Verify default to first action
        self.assertEqual(action, legal_actions[0])
        self.assertEqual(action.action_type, ActionType.DRAW)

    @pytest.mark.timeout(10)
    @patch("ollama.chat")
    async def test_get_action_api_error(self, mock_chat: Mock) -> None:
        """Test handling of API errors."""
        legal_actions: List[Action] = [
            Action(action_type=ActionType.DRAW, card=None, target=None, played_by=1),
            Action(action_type=ActionType.POINTS, card=self.p1_cards[1], target=None, played_by=1),
        ]

        # Mock API error
        mock_chat.side_effect = Exception("API Error")

        # Get AI action - should default to first legal action
        action = await self.ai_player.get_action(self.game_state, legal_actions)

        # Verify default to first action
        self.assertEqual(action, legal_actions[0])
        self.assertEqual(action.action_type, ActionType.DRAW)

    @pytest.mark.timeout(10)
    async def test_get_action_no_legal_actions(self) -> None:
        """Test handling of empty legal actions list."""
        with self.assertRaises(ValueError):
            await self.ai_player.get_action(self.game_state, [])

    def test_set_model(self) -> None:
        """Test model setting functionality."""
        test_model: str = "llama2"
        self.ai_player.set_model(test_model)
        self.assertEqual(self.ai_player.model, test_model)
