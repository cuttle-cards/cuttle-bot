"""Configuration for RL training."""
from typing import Any, Dict

# Training hyperparameters for MaskablePPO algorithm
# Using baseline config - performed best in hyperparameter search
TRAINING_CONFIG: Dict[str, Any] = {
    "total_timesteps": 500000,  # Extended training (was 100K)
    "learning_rate": 3e-4,      # Learning rate for optimizer
    "n_steps": 2048,            # Steps per update
    "batch_size": 64,           # Minibatch size
    "n_epochs": 10,             # Epochs per update
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE parameter
    "clip_range": 0.2,          # PPO clip range
    "ent_coef": 0.01,           # Entropy coefficient
    "verbose": 1,               # Logging verbosity
}

# Reward structure - optimized for self-play
# Key insight: reward scoring points, don't over-penalize turns
REWARD_CONFIG: Dict[str, float] = {
    "win": 100.0,                    # Reward for winning
    "loss": -100.0,                  # Penalty for losing
    "stalemate": -50.0,              # Penalty for stalemate (discourage draws)
    "progress_multiplier": 2.0,      # Reward for scoring points (balanced)
    "turn_penalty": -0.01,           # Keep small to not overwhelm learning
    "invalid_action_penalty": -10.0, # Penalty for illegal moves (safety check)
}

# Environment configuration
from rl.action_mapping import ACTION_SPACE_SIZE

ENV_CONFIG: Dict[str, Any] = {
    "max_actions": ACTION_SPACE_SIZE,  # Fixed action space size
    "observation_dim": 610,  # State vector dimension (136+1+180+180+4+5+52+52)
    "max_hand_size": 8,      # Max cards in hand
    "max_field_size": 10,    # Max cards on field
}

# File paths
MODEL_DIR = "rl/models"
LOG_DIR = "rl/logs"
