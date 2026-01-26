"""Gymnasium environment wrapper for Cuttle game with action masking."""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from game.card import Purpose
from game.game import Game
from rl.action_mapping import build_action_map, card_index, legal_action_mask
from rl.config import ENV_CONFIG, REWARD_CONFIG
from rl.game_logger import GameplayLogger


class CuttleRLEnvironment(gym.Env):
    """RL environment for Cuttle card game with action masking support."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, enable_logging: bool = False):
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
        self.max_steps = 300  # Increased to allow games to conclude naturally
        self.no_progress_steps = 0
        self.no_progress_limit = 60  # End early if no scoring progress
        
        # Logging
        self.logger = GameplayLogger() if enable_logging else None
        self.enable_logging = enable_logging

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
        self.no_progress_steps = 0
        
        # Reset score tracking for difference-based rewards
        self._prev_score = 0
        self._prev_opponent_score = 0
        self._prev_total_score = 0
        
        # Start logging if enabled
        if self.logger:
            self.logger.start_game(self.game)
        
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
        
        return legal_action_mask(self.game.game_state)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in environment."""
        assert self.game is not None, "Must call reset() first"
        
        # Increment step count and check for timeout
        self.step_count += 1
        if self.step_count > self.max_steps:
            print(f"⚠️  TIMEOUT: Game exceeded {self.max_steps} steps, forcing termination")
            if self.logger:
                self.logger.end_game(self.game, None, "timeout", self.step_count)
            return (
                self._encode_state(),
                -50.0,  # Strong penalty for timeout (same as stalemate)
                True,   # done
                True,   # truncated
                {"error": "timeout", "steps": self.step_count}
            )
        
        # Get current legal actions
        legal_actions = self.game.game_state.get_legal_actions()

        # Decode fixed action index into a concrete legal action
        action_map = build_action_map(legal_actions)
        chosen_action = action_map.get(action)

        # With action masking, invalid actions should never happen
        # but keep as safety check
        if chosen_action is None:
            print(f"WARNING: Invalid action {action} attempted (no matching legal action)")
            print("This should not happen with proper action masking!")
            return (
                self._encode_state(),
                REWARD_CONFIG["invalid_action_penalty"],
                True,  # done
                False, # truncated
                {"error": "invalid_action"}
            )
        
        # Log the action before execution
        if self.logger:
            self.logger.log_step(
                self.step_count,
                self.current_player,
                chosen_action,
                self.game,
                0.0,  # Reward will be updated after
                len(legal_actions)
            )
        
        turn_finished, game_ended, winner = \
            self.game.game_state.update_state(chosen_action)
        
        # Calculate reward
        reward = self._calculate_reward(game_ended, winner)

        # Track total score progress to detect stalls
        total_score = (
            self.game.game_state.get_player_score(0)
            + self.game.game_state.get_player_score(1)
        )
        if total_score > getattr(self, "_prev_total_score", 0):
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        self._prev_total_score = total_score
        
        # Update game state if turn finished
        if turn_finished:
            self.game.game_state.next_turn()
            self.current_player = (self.current_player + 1) % 2
        
        # Check if episode is done
        done = game_ended or self.game.game_state.is_stalemate()

        # Early termination if the game is stuck with no progress
        if not done and self.no_progress_steps >= self.no_progress_limit:
            print(
                f"⚠️  STALL: No scoring progress for "
                f"{self.no_progress_limit} steps, ending episode"
            )
            if self.logger:
                self.logger.end_game(self.game, None, "stall", self.step_count)
            return (
                self._encode_state(),
                REWARD_CONFIG["stalemate"],
                True,   # done
                True,   # truncated
                {"error": "stall", "steps": self.step_count}
            )
        
        # Log game end if done
        if done and self.logger:
            reason = "win" if winner is not None else "stalemate"
            self.logger.end_game(self.game, winner, reason, self.step_count)
        
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
        
        # 3. Player 0 field cards (180 dims: 10 cards × 18 dims each)
        for i in range(ENV_CONFIG["max_field_size"]):
            field = self.game.game_state.get_player_field(0)
            if i < len(field):
                card = field[i]
                obs[idx + card.suit.value[1]] = 1.0
                obs[idx + 4 + card.rank.value[1] - 1] = 1.0
                obs[idx + 17] = 1.0 if card.purpose == Purpose.POINTS else 0.0
            idx += 18
        
        # 4. Player 1 field cards (180 dims: same encoding)
        for i in range(ENV_CONFIG["max_field_size"]):
            field = self.game.game_state.get_player_field(1)
            if i < len(field):
                card = field[i]
                obs[idx + card.suit.value[1]] = 1.0
                obs[idx + 4 + card.rank.value[1] - 1] = 1.0
                obs[idx + 17] = 1.0 if card.purpose == Purpose.POINTS else 0.0
            idx += 18
        
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
        idx += 5

        # 7. Discard pile identity (52 dims)
        for card in self.game.game_state.discard_pile:
            obs[idx + card_index(card)] = 1.0
        idx += 52

        # 8. Revealed cards for seven (52 dims)
        for card in self.game.game_state.pending_seven_cards:
            obs[idx + card_index(card)] = 1.0
        idx += 52
        
        return obs

    def _calculate_reward(self, game_ended: bool, winner: Optional[int]) -> float:
        """Calculate reward for the current state.
        
        Simple reward structure focused on scoring points and winning.
        """
        if game_ended:
            if winner == self.current_player:
                return REWARD_CONFIG["win"]
            elif winner is not None:
                return REWARD_CONFIG["loss"]
            else:
                return REWARD_CONFIG["stalemate"]
        
        # Only reward our own score gains (simpler, less noisy)
        current_score = self.game.game_state.get_player_score(self.current_player)
        prev_score = getattr(self, '_prev_score', 0)
        
        score_gain = current_score - prev_score
        self._prev_score = current_score
        
        # Small reward for scoring points
        if score_gain > 0:
            return score_gain * REWARD_CONFIG["progress_multiplier"]
        
        # Minimal turn penalty otherwise
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
