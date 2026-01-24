"""Configuration for RL training."""
from typing import Any, Dict

# Training hyperparameters for MaskablePPO algorithm
TRAINING_CONFIG: Dict[str, Any] = {
    "total_timesteps": 100000,  # Total training steps
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

# Reward structure - critical for agent learning
REWARD_CONFIG: Dict[str, float] = {
    "win": 100.0,                    # Reward for winning
    "loss": -100.0,                  # Penalty for losing
    "stalemate": 0.0,                # No reward for draw
    "progress_multiplier": 10.0,     # Multiplier for score progress
    "turn_penalty": -1.0,            # Small penalty each turn
    "invalid_action_penalty": -50.0, # Heavy penalty for illegal moves (safety check)
}

# Environment configuration
ENV_CONFIG: Dict[str, Any] = {
    "max_actions": 50,       # Max possible actions per turn
    "observation_dim": 206,  # State vector dimension (136+1+30+30+4+5)
    "max_hand_size": 8,      # Max cards in hand
    "max_field_size": 10,    # Max cards on field
}

# File paths
MODEL_DIR = "rl/models"
LOG_DIR = "rl/logs"
