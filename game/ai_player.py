"""
AI player module for the Cuttle card game.

This module provides the AIPlayer class that uses the Ollama language model
to make strategic decisions in the game. The AI player understands game rules,
implements strategies, and makes decisions based on the current game state.
"""

from __future__ import annotations

import time
from typing import List

import ollama

from game.action import Action
from game.card import Card, Purpose
from game.game_state import GameState
from game.utils import log_print


class AIPlayer:
    """AI player that uses Ollama LLM to make decisions in the game.

    This class implements an AI player that uses a large language model (LLM)
    to analyze game states and make strategic decisions. It understands game rules,
    implements various strategies, and tries to avoid common mistakes.

    The AI player can:
    - Analyze the current game state
    - Choose optimal actions from available legal moves
    - Make strategic decisions about card usage
    - Handle special card effects and combinations

    Attributes:
        model (str): The Ollama model to use for decision making.
        max_retries (int): Maximum number of retries for failed LLM calls.
        retry_delay (int): Delay in seconds between retries.
        GAME_CONTEXT (str): Detailed game rules and strategy guide for the LLM.
    """

    # Game rules and strategy context for the LLM
    GAME_CONTEXT = """
You are an expert of playing competitive card games. You excel at reasoning through the rules of the card game and making optimal decisions. You are great at identifying patterns and making strategic moves. You are playing a card game called Cuttle. Here are the key rules and strategies:

Rules:
1. Win condition: Reach your point target (initial target is 21 points)
2. Card Actions:
    - Play as points (number cards 1-10), only Ace through Ten are counted as points. Eight played as face card is not counted as points.
    - Play as face cards (Kings, Queens, Jacks, Eights)
    - Play as one-off effects (Aces, Threes, Fours, Fives, Sixes)
        - Aces clears all point cards for both players
        - Threes let's you choose a card from the scrap pile. Avoid playing Threes as one-off when the scrap pile is empty or does not have any cards you want.
        - Fives will let you draw the top two cards from the deck.
        - Sixes clears all face cards for both players
    - Scuttle: Play a higher point card to destroy opponent's point card
    - Counter: Use a Two to counter any one-off effect
3. Kings reduce your target score (1 King: 14, 2 Kings: 10, 3 Kings: 5, 4 Kings: 0)
4. Face cards provide special abilities. Face cards are not counted as points.
    - King: Reduces target score
    - Queen: Protects your points from face cards, certain targeted one-offs, and counters. Does not protect against Ace One-offs or Six One-offs.
    - Jack: Steals opponent's points
    - Eight: Glasses (opponent plays with revealed hand)


Strategies:
1. Optimize to increase your score and decrease your target score. If you have a high value point card, try to play it as points. If a move increases your score to meet or exceed your target score, do it straight away.
2. Prioritize playing Kings early to reduce your target score
3. Save Twos for countering important one-off effects. Favor drawing a card over playing a two as points.
4. Use Jacks to steal high-value point cards
5. Protect high-value points with Queens
6. Use Aces to clear opponent's strong point cards. Avoid playing Aces as one-off when opponent doesn't have any point cards on field. Avoid playing Aces as points when possible since the reward is low.
7. Keep track of used Twos to know when one-offs are safe
8. Scuttle opponent's high-value points when possible
9. Avoid playing Six as one-off when opponent doesn't have any face cards on field.
10. If opponent score is close to opponent's target, try to play Aces as one-off to clear their points, or play Sixes as one-off to clear their Kings if any are on field.

Mistakes to avoid:
1. Playing Aces as one-off when opponent doesn't have any point cards on field.
2. Playing Aces as points since the reward is low.
3. Playing Threes as one-off when the scrap pile is empty or does not have any cards you want.
4. Playing Sixes as one-off when opponent doesn't have any face cards on field.

The Strategy is key to winning the game.
    """

    def __init__(self):
        """Initialize the AI player.

        Sets up:
        - The default language model (gemma3:4b)
        - Retry settings for LLM calls
        - Verifies AI's understanding of game rules
        """
        self.model = "gemma3:4b"  # Default to mistral model
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        # Initialize system context and verify AI understanding
        self._verify_ai_understanding()

    def _verify_ai_understanding(self):
        """Verify that the AI understands the game rules and strategies.

        This method sends a test prompt to the LLM to confirm it understands
        the game rules and strategies. The response is logged for debugging.

        Note:
            If verification fails, a warning is logged but the AI can still function.
        """
        verification_prompt = """Please confirm that you understand the game rules and strategies for Cuttle.
Respond with a brief summary of your understanding and confirm that you will avoid the common mistakes listed.
Keep your response concise."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.GAME_CONTEXT},
                    {"role": "user", "content": verification_prompt},
                ],
            )
            log_print("AI Understanding Verification:")
            log_print(response.message.content)
        except Exception as e:
            log_print(f"Warning: Could not verify AI understanding: {e}")

    def _format_game_state(
        self,
        game_state: GameState,
        legal_actions: List[Action],
        is_human_view: bool = False,
    ) -> str:
        """Format the current game state and legal actions into a prompt for the LLM.

        This method creates a detailed prompt that includes:
        - Current state of both players (hand, field, score, target)
        - Deck and discard pile information
        - List of legal actions
        - Instructions for the LLM

        Args:
            game_state (GameState): The current state of the game.
            legal_actions (List[Action]): List of legal actions available.
            is_human_view (bool, optional): Whether to hide AI's hand. Defaults to False.

        Returns:
            str: Formatted prompt string for the LLM.
        """
        opponent_point_cards = [
            card for card in game_state.fields[0] if card.purpose == Purpose.POINTS
        ]
        opponent_face_cards = [
            card for card in game_state.fields[0] if card.purpose == Purpose.FACE_CARD
        ]

        legal_actions_str = "\n".join(
            [f"- action {i}: {action}" for i, action in enumerate(legal_actions)]
        )

        prompt = f"""
Current Game State:
AI
{"AI Hand: " + str(game_state.hands[1]) if not is_human_view else "AI Hand: [Hidden]"}
AI Field: {game_state.fields[1]}
AI Score: {game_state.get_player_score(1)}
AI Target: {game_state.get_player_target(1)}

Opponent
Opponent's Hand Size: {len(game_state.hands[0])}
Opponent's Field: {game_state.fields[0]}
Opponent's Point Cards: {opponent_point_cards}
Opponent's Face Cards: {opponent_face_cards}
Opponent's Score: {game_state.get_player_score(0)}
Opponent's Target: {game_state.get_player_target(0)}

Deck Size: {len(game_state.deck)}
Discard Pile Size: {len(game_state.discard_pile)}

Legal Actions:
{legal_actions_str}

Instructions:
1. Analyze the game state and available actions. What is opponent's score and target? What is your score and target?
2. Choose the best action among the legal actions based on the game rules and strategies, think through the consequences of your actions and a few turns ahead. Avoid making mistakes that will cost you the game.
3. IMPORTANT: Your response MUST include a valid action number from the list above
4. Stop thinking and make a choice after a few seconds.
5. If there is only one action, choose it without thinking.
6. Action number should be a number from 0 to {len(legal_actions) - 1}
6. Format your response as:
    Reasoning: [brief explanation]
    Choice: [action number]

Make your choice now:
        """
        return prompt

    async def get_action(
        self, game_state: GameState, legal_actions: List[Action]
    ) -> Action:
        """Get the AI's chosen action based on the current game state.

        This method:
        1. Validates that legal actions are available
        2. Formats the game state into a prompt
        3. Sends the prompt to the LLM
        4. Extracts and validates the chosen action
        5. Retries on failure up to max_retries times

        Args:
            game_state (GameState): The current state of the game.
            legal_actions (List[Action]): List of legal actions available.

        Returns:
            Action: The chosen action to perform.

        Raises:
            ValueError: If no legal actions are available.

        Note:
            If all retries fail, returns the first legal action as a fallback.
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Format the game state and actions into a prompt
        prompt = self._format_game_state(game_state, legal_actions)
        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Get response from Ollama with system context
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.GAME_CONTEXT},
                        {"role": "user", "content": prompt},
                    ],
                )

                # Extract the action number from the response
                log_print(response)
                response_text = response.message.content
                log_print(response_text)
                # Look for "Choice: [number]" pattern first
                import re

                choice_match = re.search(r"Choice:\s*(\d+)", response_text)
                if choice_match:
                    action_idx = int(choice_match.group(1))
                else:
                    # Fallback to finding any number in the response
                    numbers = re.findall(r"\d+", response_text)
                    if not numbers:
                        raise ValueError("No action number found in response")
                    action_idx = int(numbers[-1])

                # Validate the action index
                if action_idx < 0 or action_idx >= len(legal_actions):
                    raise ValueError(f"Invalid action index: {action_idx}")

                return legal_actions[action_idx]

            except Exception as e:
                last_error = e
                print(
                    f"Error getting AI action (attempt {retries + 1}/{self.max_retries}): {e}"
                )
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

        print(f"All retries failed. Using first legal action. Last error: {last_error}")
        return legal_actions[0]

    def set_model(self, model: str):
        """Set the model to use for AI decisions."""
        self.model = model

    def choose_card_from_discard(self, discard_pile: List[Card]) -> Card:
        """Choose a card from the discard pile when playing a Three."""
        # Format the prompt for the LLM
        prompt = f"""
        You need to choose a card from the discard pile. Here are the available cards:
        {[str(card) for card in discard_pile]}

        Consider these factors when choosing:
        1. High point cards (7-10) are valuable for scoring
        2. Face cards (Kings, Queens, Jacks) provide powerful effects
        3. Twos are valuable for countering one-offs
        4. One-off cards (Aces, Threes, Fives, Sixes) can be useful for special effects

        Instructions:
        1. Analyze the available cards
        2. Choose the most valuable card based on the game rules and strategies
        3. IMPORTANT: Your response MUST be a number from 0 to {len(discard_pile) - 1}
        4. Format your response as:
           Reasoning: [brief explanation]
           Choice: [index number]

        Make your choice now:
        """

        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Get response from Ollama with system context
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.GAME_CONTEXT},
                        {"role": "user", "content": prompt},
                    ],
                )

                # Extract the action number from the response
                response_text = response.message.content
                print("AI response:", response_text)

                # Look for "Choice: [number]" pattern first
                import re

                choice_match = re.search(r"Choice:\s*(\d+)", response_text)
                if choice_match:
                    card_idx = int(choice_match.group(1))
                else:
                    # Fallback to finding any number in the response
                    numbers = re.findall(r"\d+", response_text)
                    if not numbers:
                        raise ValueError("No card index found in response")
                    card_idx = int(numbers[-1])

                # Validate the card index
                if card_idx < 0 or card_idx >= len(discard_pile):
                    raise ValueError(f"Invalid card index: {card_idx}")

                return discard_pile[card_idx]

            except Exception as e:
                last_error = e
                print(
                    f"Error choosing card from discard (attempt {retries + 1}/{self.max_retries}): {e}"
                )
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

        print(
            f"All retries failed. Using first card from discard pile. Last error: {last_error}"
        )
        return discard_pile[0]

    def choose_two_cards_from_hand(self, hand: List[Card]) -> List[Card]:
        """Choose up to two cards to discard from hand when affected by a Four one-off effect."""
        # Format the prompt for the LLM
        prompt = f"""
        You need to choose up to two cards to discard from your hand. Here are the available cards:
        {[str(card) for card in hand]}

        Consider these factors when choosing:
        1. Prioritize discarding low-value point cards (1-6)
        2. Avoid discarding valuable cards like:
           - High point cards (7-10)
           - Face cards (Kings, Queens, Jacks)
           - Twos (valuable for countering one-offs)
           - One-off cards (Aces, Threes, Fives, Sixes)
        3. If you have multiple low-value cards, choose the ones with the lowest point values

        Instructions:
        1. Analyze the available cards
        2. Choose up to two cards to discard based on the game rules and strategies
        3. IMPORTANT: Your response MUST be a comma-separated list of numbers from 0 to {len(hand) - 1}
        4. Format your response as:
           Reasoning: [brief explanation]
           Choice: [index1, index2]

        Make your choice now:
        """

        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Get response from Ollama with system context
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.GAME_CONTEXT},
                        {"role": "user", "content": prompt},
                    ],
                )

                # Extract the card indices from the response
                response_text = response.message.content
                print("AI response:", response_text)

                # Look for "Choice: [number1, number2]" pattern first
                import re

                choice_match = re.search(r"Choice:\s*\[([\d,\s]+)\]", response_text)
                if choice_match:
                    indices_str = choice_match.group(1)
                    card_indices = [
                        int(idx.strip())
                        for idx in indices_str.split(",")
                        if idx.strip()
                    ]
                else:
                    # Fallback to finding any numbers in the response
                    numbers = re.findall(r"\d+", response_text)
                    if not numbers:
                        raise ValueError("No card indices found in response")
                    # Take up to 2 numbers from the response
                    card_indices = [int(num) for num in numbers[:2]]

                # Validate the card indices
                valid_indices = [idx for idx in card_indices if 0 <= idx < len(hand)]
                if not valid_indices:
                    raise ValueError(f"No valid card indices found: {card_indices}")

                # Return up to 2 cards
                return [hand[idx] for idx in valid_indices[:2]]

            except Exception as e:
                last_error = e
                print(
                    f"Error choosing cards from hand (attempt {retries + 1}/{self.max_retries}): {e}"
                )
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

        print(
            f"All retries failed. Using first two cards from hand. Last error: {last_error}"
        )
        # Return up to 2 cards from the hand
        return hand[: min(2, len(hand))]
