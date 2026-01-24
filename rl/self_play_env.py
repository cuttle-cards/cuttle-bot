"""Self-play wrapper with action masking support."""
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from rl.cuttle_env import CuttleRLEnvironment


class SelfPlayWrapper(gym.Wrapper):
    """Wrapper that enables self-play training with action masking."""
    
    def __init__(self, env: CuttleRLEnvironment):
        super().__init__(env)
        self.opponent_policy = "random"  # Strategy: "random" or future: "model"
    
    def action_masks(self) -> np.ndarray:
        """Forward action masks from wrapped environment."""
        return self.env.action_masks()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute agent's action, then opponent's action (both use masking)."""
        # Agent's move
        obs, reward, done, truncated, info = self.env.step(action)
        
        if done:
            return obs, reward, done, truncated, info
        
        # Opponent's turn with action masking
        opponent_mask = self.env.action_masks()
        opponent_legal_indices = np.where(opponent_mask)[0]
        
        if len(opponent_legal_indices) > 0:
            # Random opponent chooses from legal actions only
            opponent_action = np.random.choice(opponent_legal_indices)
            obs, opp_reward, done, truncated, info = self.env.step(opponent_action)
            
            # Flip reward: opponent's loss is agent's gain
            reward = -opp_reward
        
        return obs, reward, done, truncated, info
