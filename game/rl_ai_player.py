"""
RL AI player module for the Cuttle card game.

This module provides the RLAIPlayer class that uses a trained reinforcement learning
model to make strategic decisions in the game. It integrates with the existing game
system and can be used as a drop-in replacement for the LLM-based AIPlayer.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from sb3_contrib import MaskablePPO

from game.action import Action
from game.card import Card, Rank
from game.game_state import GameState
from rl.cuttle_env import CuttleRLEnvironment
from rl.self_play_env import SelfPlayWrapper


class RLAIPlayer:
    """RL-based AI player that uses a trained reinforcement learning model.
    
    This class implements an AI player that uses a trained MaskablePPO model
    to make strategic decisions. It integrates with the existing game system
    and provides the same interface as the LLM-based AIPlayer.
    
    The RL AI player can:
    - Load a trained RL model
    - Convert game states to RL observations
    - Use action masking to ensure legal moves
    - Make decisions based on learned strategies
    
    Attributes:
        model_path (str): Path to the trained RL model.
        model: The loaded MaskablePPO model.
        env: The RL environment for state encoding.
        max_retries (int): Maximum number of retries for failed predictions.
        retry_delay (float): Delay in seconds between retries.
    """
    
    def __init__(
        self, 
        model_path: str = "rl/models/cuttle_rl_final",
        max_retries: int = 3,
        retry_delay: float = 0.1
    ):
        """Initialize the RL AI player.
        
        Args:
            model_path (str): Path to the trained RL model (without .zip extension).
            max_retries (int): Maximum number of retries for failed predictions.
            retry_delay (float): Delay in seconds between retries.
        """
        self.model_path = model_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize environment for state encoding
        self.env = CuttleRLEnvironment()
        self.env = SelfPlayWrapper(self.env)
        
        # Load the trained model
        self.model = self._load_model()
        
    def _load_model(self) -> MaskablePPO:
        """Load the trained RL model.
        
        Returns:
            MaskablePPO: The loaded model.
            
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            Exception: If model loading fails.
        """
        model_file = f"{self.model_path}.zip"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            model = MaskablePPO.load(self.model_path, env=self.env)
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def _encode_game_state(self, game_state: GameState) -> np.ndarray:
        """Encode the game state as an RL observation.
        
        Args:
            game_state (GameState): The current game state.
            
        Returns:
            np.ndarray: Encoded observation vector.
        """
        # Create a temporary game instance with the current state
        from game.game import Game
        temp_game = Game()
        temp_game.game_state = game_state
        
        # Set the game state in the environment
        self.env.env.unwrapped.game = temp_game
        
        # Encode the state using the underlying environment
        return self.env.env.unwrapped._encode_state()
    
    def _get_action_mask(self, legal_actions: List[Action]) -> np.ndarray:
        """Get action mask for the current legal actions."""
        from rl.action_mapping import legal_action_mask_from_actions

        return legal_action_mask_from_actions(legal_actions)
    
    async def get_action(
        self, 
        game_state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """Get the RL AI's chosen action based on the current game state.
        
        This method:
        1. Validates that legal actions are available
        2. Encodes the game state as an RL observation
        3. Uses the trained model to predict the best action
        4. Maps the predicted action index to the actual Action object
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
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Encode the game state
                observation = self._encode_game_state(game_state)
                
                # Get action mask
                action_mask = self._get_action_mask(legal_actions)
                
                # Predict action using the model
                action_index, _ = self.model.predict(
                    observation, 
                    action_masks=action_mask, 
                    deterministic=True
                )
                
                action_index = int(action_index)

                from rl.action_mapping import build_action_map

                action_map = build_action_map(legal_actions)
                if action_index not in action_map:
                    return legal_actions[0]
                return action_map[action_index]
                
            except Exception as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    import time
                    time.sleep(self.retry_delay)
        
        # If all retries failed, return the first legal action as fallback
        print(f"RL AI fallback: All {self.max_retries} retries failed. Last error: {last_error}")
        return legal_actions[0]
    
    def get_action_sync(
        self, 
        game_state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """Synchronous version of get_action for non-async contexts.
        
        Args:
            game_state (GameState): The current state of the game.
            legal_actions (List[Action]): List of legal actions available.
            
        Returns:
            Action: The chosen action to perform.
        """
        import asyncio

        # Run the async method in a new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new one
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_action(game_state, legal_actions))
                    return future.result()
            else:
                return loop.run_until_complete(self.get_action(game_state, legal_actions))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.get_action(game_state, legal_actions))
    
    def choose_card_from_discard(self, discard_pile: List[Card]) -> Card:
        """Choose a card from the discard pile when playing a Three.
        
        Args:
            discard_pile (List[Card]): Available cards in the discard pile.
            
        Returns:
            Card: The chosen card.
        """
        if not discard_pile:
            raise ValueError("No cards in discard pile")
        
        # Simple strategy: choose the highest point value card
        # Prioritize high point cards (7-10), then face cards, then others
        def card_value(card: Card) -> int:
            if card.point_value() <= 10:  # Point cards
                return card.point_value() + 100  # High priority for point cards
            elif card.is_face_card():  # Face cards
                return 50 + card.point_value()  # Medium priority
            else:  # One-off cards
                return card.point_value()  # Lower priority
        
        # Sort by value (highest first) and choose the best card
        best_card = max(discard_pile, key=card_value)
        return best_card
    
    def choose_two_cards_from_hand(self, hand: List[Card]) -> List[Card]:
        """Choose up to two cards to discard from hand when affected by a Four one-off effect.
        
        Args:
            hand (List[Card]): Available cards in the hand.
            
        Returns:
            List[Card]: Up to two cards to discard.
        """
        if not hand:
            return []
        
        # Simple strategy: discard the lowest value cards
        # Prioritize keeping high point cards, face cards, and Twos
        def card_priority(card: Card) -> int:
            if card.point_value() <= 10:  # Point cards
                return card.point_value()  # Lower point value = lower priority (discard first)
            elif card.is_face_card():  # Face cards
                return 100  # High priority (keep)
            elif card.rank == Rank.TWO:  # Twos are valuable for countering
                return 90  # High priority (keep)
            else:  # One-off cards
                return 50  # Medium priority
        
        # Sort by priority (lowest first) and take up to 2 cards
        sorted_hand = sorted(hand, key=card_priority)
        return sorted_hand[:min(2, len(sorted_hand))]


class RLAIPlayerWrapper:
    """Wrapper to make RLAIPlayer compatible with existing AIPlayer interface.
    
    This wrapper provides the same interface as the original AIPlayer
    but uses the RL model for decision making.
    """
    
    def __init__(self, model_path: str = "rl/models/cuttle_rl_final"):
        """Initialize the RL AI player wrapper.
        
        Args:
            model_path (str): Path to the trained RL model.
        """
        self.rl_ai = RLAIPlayer(model_path)
        self.model = "rl_model"  # For compatibility
        self.max_retries = 3
        self.retry_delay = 0.1
    
    async def get_action(
        self, 
        game_state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """Get action using the RL model.
        
        Args:
            game_state (GameState): The current state of the game.
            legal_actions (List[Action]): List of legal actions available.
            
        Returns:
            Action: The chosen action to perform.
        """
        return await self.rl_ai.get_action(game_state, legal_actions)
    
    def get_action_sync(
        self, 
        game_state: GameState, 
        legal_actions: List[Action]
    ) -> Action:
        """Synchronous version of get_action.
        
        Args:
            game_state (GameState): The current state of the game.
            legal_actions (List[Action]): List of legal actions available.
            
        Returns:
            Action: The chosen action to perform.
        """
        return self.rl_ai.get_action_sync(game_state, legal_actions)
    
    def choose_card_from_discard(self, discard_pile: List[Card]) -> Card:
        """Choose a card from the discard pile when playing a Three.
        
        Args:
            discard_pile (List[Card]): Available cards in the discard pile.
            
        Returns:
            Card: The chosen card.
        """
        return self.rl_ai.choose_card_from_discard(discard_pile)
    
    def choose_two_cards_from_hand(self, hand: List[Card]) -> List[Card]:
        """Choose up to two cards to discard from hand when affected by a Four one-off effect.
        
        Args:
            hand (List[Card]): Available cards in the hand.
            
        Returns:
            List[Card]: Up to two cards to discard.
        """
        return self.rl_ai.choose_two_cards_from_hand(hand)
