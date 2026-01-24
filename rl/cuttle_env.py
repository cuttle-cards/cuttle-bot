"""Gymnasium environment wrapper for Cuttle game with action masking."""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from game.card import Purpose
from game.game import Game
from rl.config import ENV_CONFIG, REWARD_CONFIG


class CuttleRLEnvironment(gym.Env):
    """RL environment for Cuttle card game with action masking support."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(ENV_CONFIG["max_actions"])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ENV_CONFIG["observation_dim"],),
            dtype=np.float32
        )
        
        # Game instance
        self.game: Optional[Game] = None
        self.current_player = 0
        self.step_count = 0
        self.max_steps = 200  # Add timeout to prevent infinite loops

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize new game without AI player
        self.game = Game(manual_selection=False, ai_player=None)
        self.current_player = 0
        self.step_count = 0
        
        # Get initial observation
        observation = self._encode_state()
        info = self._get_info()
        
        return observation, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions.
        
        This is the key method for action masking. It returns a boolean array
        where True indicates a legal action and False indicates illegal.
        
        Returns:
            np.ndarray: Boolean mask of shape (max_actions,)
        """
        assert self.game is not None, "Must call reset() first"
        
        # Get current legal actions
        legal_actions = self.game.game_state.get_legal_actions()
        
        # Create mask: True for legal actions, False for illegal
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        mask[:len(legal_actions)] = True
        
        return mask

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in environment."""
        assert self.game is not None, "Must call reset() first"
        
        # Increment step count and check for timeout
        self.step_count += 1
        if self.step_count > self.max_steps:
            print(f"⚠️  TIMEOUT: Game exceeded {self.max_steps} steps, forcing termination")
            return (
                self._encode_state(),
                -10.0,  # Penalty for timeout
                True,   # done
                True,   # truncated
                {"error": "timeout", "steps": self.step_count}
            )
        
        # Get current legal actions
        legal_actions = self.game.game_state.get_legal_actions()
        
        # With action masking, invalid actions should never happen
        # but keep as safety check
        if action >= len(legal_actions):
            print(f"WARNING: Invalid action {action} attempted (max: {len(legal_actions)-1})")
            print("This should not happen with proper action masking!")
            return (
                self._encode_state(),
                REWARD_CONFIG["invalid_action_penalty"],
                True,  # done
                False, # truncated
                {"error": "invalid_action"}
            )
        
        # Execute the chosen action
        chosen_action = legal_actions[action]
        turn_finished, game_ended, winner = \
            self.game.game_state.update_state(chosen_action)
        
        # Calculate reward
        reward = self._calculate_reward(game_ended, winner)
        
        # Update game state if turn finished
        if turn_finished:
            self.game.game_state.next_turn()
            self.current_player = (self.current_player + 1) % 2
        
        # Check if episode is done
        done = game_ended or self.game.game_state.is_stalemate()
        
        # Get new observation
        observation = self._encode_state()
        info = self._get_info()
        
        return observation, reward, done, False, info

    def _encode_state(self) -> np.ndarray:
        """Encode game state as fixed-size vector."""
        assert self.game is not None
        
        obs = np.zeros(ENV_CONFIG["observation_dim"], dtype=np.float32)
        idx = 0
        
        # 1. Current player's hand (136 dims: 8 cards × 17 dims each)
        hand = self.game.game_state.hands[self.current_player]
        for i in range(ENV_CONFIG["max_hand_size"]):
            if i < len(hand):
                card = hand[i]
                obs[idx + card.suit.value[1]] = 1.0
                obs[idx + 4 + card.rank.value[1] - 1] = 1.0
            idx += 17
        
        # 2. Opponent hand size (1 dim, normalized)
        opponent = 1 - self.current_player
        obs[idx] = len(self.game.game_state.hands[opponent]) / 8.0
        idx += 1
        
        # 3. Player 0 field cards (30 dims: 10 cards × 3 dims each)
        for i in range(ENV_CONFIG["max_field_size"]):
            field = self.game.game_state.get_player_field(0)
            if i < len(field):
                card = field[i]
                obs[idx] = 1.0
                obs[idx + 1] = card.rank.value[1] / 13.0
                obs[idx + 2] = 1.0 if card.purpose == Purpose.POINTS else 0.0
            idx += 3
        
        # 4. Player 1 field cards (30 dims: same encoding)
        for i in range(ENV_CONFIG["max_field_size"]):
            field = self.game.game_state.get_player_field(1)
            if i < len(field):
                card = field[i]
                obs[idx] = 1.0
                obs[idx + 1] = card.rank.value[1] / 13.0
                obs[idx + 2] = 1.0 if card.purpose == Purpose.POINTS else 0.0
            idx += 3
        
        # 5. Scores and targets (4 dims)
        obs[idx] = self.game.game_state.get_player_score(0) / 21.0
        obs[idx + 1] = self.game.game_state.get_player_score(1) / 21.0
        obs[idx + 2] = self.game.game_state.get_player_target(0) / 21.0
        obs[idx + 3] = self.game.game_state.get_player_target(1) / 21.0
        idx += 4
        
        # 6. Game state flags (5 dims)
        obs[idx] = float(self.current_player)
        obs[idx + 1] = 1.0 if self.game.game_state.resolving_one_off else 0.0
        obs[idx + 2] = 1.0 if self.game.game_state.resolving_three else 0.0
        obs[idx + 3] = len(self.game.game_state.deck) / 52.0
        obs[idx + 4] = len(self.game.game_state.discard_pile) / 52.0
        
        return obs

    def _calculate_reward(self, game_ended: bool, winner: Optional[int]) -> float:
        """Calculate reward for the current state."""
        if game_ended:
            if winner == self.current_player:
                return REWARD_CONFIG["win"]
            elif winner is not None:
                return REWARD_CONFIG["loss"]
            else:
                return REWARD_CONFIG["stalemate"]
        
        # Intermediate reward: progress toward target
        current_score = self.game.game_state.get_player_score(self.current_player)
        target = self.game.game_state.get_player_target(self.current_player)
        
        if target > 0:
            progress = current_score / target
            return (progress * REWARD_CONFIG["progress_multiplier"] + 
                    REWARD_CONFIG["turn_penalty"])
        else:
            return REWARD_CONFIG["turn_penalty"]
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about game state."""
        assert self.game is not None
        return {
            "current_player": self.current_player,
            "legal_actions": len(self.game.game_state.get_legal_actions()),
            "player_0_score": self.game.game_state.get_player_score(0),
            "player_1_score": self.game.game_state.get_player_score(1),
        }
