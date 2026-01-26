"""Self-play wrapper with action masking support and model-based opponent."""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from rl.cuttle_env import CuttleRLEnvironment


class SelfPlayWrapper(gym.Wrapper):
    """Wrapper that enables self-play training with action masking.
    
    Supports two opponent modes:
    - "random": Opponent chooses randomly from legal actions (default for early training)
    - "model": Opponent uses the trained model (true self-play)
    """
    
    def __init__(
        self, 
        env: CuttleRLEnvironment,
        opponent_type: str = "random",
    ):
        super().__init__(env)
        self.opponent_type = opponent_type
        self._opponent_model = None
        self._update_freq = 1000  # Update opponent model every N steps
        self._steps_since_update = 0
        
    def set_opponent_model(self, model) -> None:
        """Set the model to use for opponent actions.
        
        Args:
            model: A trained MaskablePPO model (or compatible)
        """
        self._opponent_model = model
        self.opponent_type = "model"
        
    def action_masks(self) -> np.ndarray:
        """Forward action masks from wrapped environment."""
        return self.env.action_masks()
    
    def _get_opponent_action(self, mask: np.ndarray) -> int:
        """Get opponent's action based on opponent_type."""
        legal_indices = np.where(mask)[0]
        
        if len(legal_indices) == 0:
            return 0  # Fallback (shouldn't happen with proper masking)
        
        if self.opponent_type == "model" and self._opponent_model is not None:
            # Use the model to predict action
            obs = self.env._encode_state()
            try:
                action, _ = self._opponent_model.predict(
                    obs,
                    deterministic=False,  # Add some exploration
                    action_masks=mask,
                )
                return int(action)
            except Exception:
                # Fallback to random if prediction fails
                return int(np.random.choice(legal_indices))
        else:
            # Random opponent (default)
            return int(np.random.choice(legal_indices))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute agent's action, then opponent's action (both use masking)."""
        # Agent's move
        obs, reward, done, truncated, info = self.env.step(action)
        
        if done:
            return obs, reward, done, truncated, info
        
        # Opponent's turn with action masking
        opponent_mask = self.env.action_masks()
        opponent_action = self._get_opponent_action(opponent_mask)
        
        obs, opp_reward, done, truncated, info = self.env.step(opponent_action)
        
        # Flip reward: opponent's loss is agent's gain
        reward = -opp_reward
        
        self._steps_since_update += 1
        
        return obs, reward, done, truncated, info


class AdaptiveSelfPlayWrapper(SelfPlayWrapper):
    """Self-play wrapper that gradually transitions from random to model opponent.
    
    Starts with random opponent and progressively increases model usage
    based on training progress.
    """
    
    def __init__(
        self,
        env: CuttleRLEnvironment,
        model_prob_start: float = 0.0,
        model_prob_end: float = 0.8,
        transition_steps: int = 100000,
    ):
        super().__init__(env, opponent_type="adaptive")
        self.model_prob_start = model_prob_start
        self.model_prob_end = model_prob_end
        self.transition_steps = transition_steps
        self._total_steps = 0
        
    def _get_model_probability(self) -> float:
        """Get current probability of using model opponent."""
        if self._opponent_model is None:
            return 0.0
        progress = min(1.0, self._total_steps / self.transition_steps)
        return self.model_prob_start + progress * (self.model_prob_end - self.model_prob_start)
    
    def _get_opponent_action(self, mask: np.ndarray) -> int:
        """Get opponent action, mixing random and model based on progress."""
        legal_indices = np.where(mask)[0]
        
        if len(legal_indices) == 0:
            return 0
        
        # Decide whether to use model or random
        use_model = (
            self._opponent_model is not None 
            and np.random.random() < self._get_model_probability()
        )
        
        if use_model:
            obs = self.env._encode_state()
            try:
                action, _ = self._opponent_model.predict(
                    obs,
                    deterministic=False,
                    action_masks=mask,
                )
                return int(action)
            except Exception:
                return int(np.random.choice(legal_indices))
        else:
            return int(np.random.choice(legal_indices))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step and track total steps for adaptive scheduling."""
        self._total_steps += 1
        return super().step(action)
