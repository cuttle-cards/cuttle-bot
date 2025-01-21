from __future__ import annotations
import ollama
from typing import List, Dict, Any
import time
from game.action import Action, ActionType
from game.game_state import GameState
from game.card import Card, Purpose


class AIPlayer:
    """AI player that uses Ollama LLM to make decisions in the game."""

    # Game rules and strategy context for the LLM
    GAME_CONTEXT = """
    You are playing a card game called Cuttle. Here are the key rules and strategies:

    Rules:
    1. Win condition: Reach your point target (starts at 21, reduced by Kings)
    2. Card Actions:
        - Play as points (number cards 2-10)
        - Play as face cards (Kings, Queens, Jacks, Eights)
        - Play as one-off effects (Aces, Threes, Fours, Fives, Sixes)
        - Scuttle: Play a higher point card to destroy opponent's point card
        - Counter: Use a Two to counter any one-off effect
    3. Kings reduce your target score (1 King: 14, 2 Kings: 10, 3 Kings: 5, 4 Kings: 0)
    4. Face cards provide special abilities:
        - King: Reduces target score
        - Queen: Protects your points from scuttling
        - Jack: Steals opponent's points
        - Eight: Glasses (opponent plays with revealed hand)

    Strategies:
    1. Prioritize playing Kings early to reduce your target score
    2. Save Twos for countering important one-off effects
    3. Use Jacks to steal high-value point cards
    4. Protect high-value points with Queens
    5. Use Aces to clear opponent's strong point cards
    6. Consider opponent's possible counters before playing one-offs
    7. Keep track of used Twos to know when one-offs are safe
    8. Scuttle opponent's high-value points when possible
    """

    def __init__(self):
        """Initialize the AI player."""
        self.model = "llama3.2"  # Default to mistral model
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def _format_game_state(
        self,
        game_state: GameState,
        legal_actions: List[Action],
        is_human_view: bool = False,
    ) -> str:
        """Format the current game state and legal actions into a prompt for the LLM."""
        prompt = f"""
        Current Game State:
        {'Your Hand (P1): ' + str(game_state.hands[1]) if not is_human_view else 'AI Hand: [Hidden]'}
        AI Field: {game_state.fields[1]}
        Opponent's Hand Size: {len(game_state.hands[0])}
        Opponent's Field: {game_state.fields[0]}
        Deck Size: {len(game_state.deck)}
        Discard Pile Size: {len(game_state.discard_pile)}
        AI Score: {game_state.get_player_score(1)}
        AI Target: {game_state.get_player_target(1)}
        Opponent's Score: {game_state.get_player_score(0)}
        Opponent's Target: {game_state.get_player_target(0)}

        Legal Actions:
        {[f"{i}: {action}" for i, action in enumerate(legal_actions)]}

        Instructions:
        1. Analyze the game state and available actions
        2. Choose the best action based on the game rules and strategies
        3. IMPORTANT: Your response MUST include a valid action number from the list above
        4. Format your response as:
           Reasoning: [brief explanation]
           Choice: [action number]

        Make your choice now:
        """
        return prompt

    async def get_action(
        self, game_state: GameState, legal_actions: List[Action]
    ) -> Action:
        """Get the AI's chosen action based on the current game state."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Format the game state and actions into a prompt
        prompt = self._format_game_state(game_state, legal_actions)

        # Add game context and strategies
        full_prompt = self.GAME_CONTEXT + "\n" + prompt

        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # Get response from Ollama
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                )

                # Extract the action number from the response
                response_text = response.message.content

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

    def set_model(self, model_name: str) -> None:
        """Set the model to use for chat completion."""
        self.model = model_name
