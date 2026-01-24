"""Evaluate trained RL agent with action masking."""
import os
from typing import Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from rl.config import MODEL_DIR
from rl.cuttle_env import CuttleRLEnvironment


def play_episode(
    model: MaskablePPO, 
    env: CuttleRLEnvironment, 
    deterministic: bool = True
) -> Tuple[float, int, Optional[int]]:
    """Play one episode with action masking."""
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    
    while not done:
        # Agent's turn with action mask
        action_mask = env.action_masks()
        action, _ = model.predict(
            obs, 
            action_masks=action_mask,  # Pass mask to model
            deterministic=deterministic
        )
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        if done:
            break
        
        # Random opponent's turn (also uses masking)
        opponent_mask = env.action_masks()
        legal_indices = np.where(opponent_mask)[0]
        if len(legal_indices) > 0:
            opp_action = np.random.choice(legal_indices)
            obs, opp_reward, done, truncated, info = env.step(opp_action)
            episode_reward -= opp_reward
            steps += 1
    
    # Get winner
    winner = env.game.game_state.winner() if env.game else None
    
    return episode_reward, steps, winner


def evaluate_agent(model_path: str, n_episodes: int = 100):
    """Evaluate agent over multiple episodes."""
    print(f"Loading MaskablePPO model from: {model_path}")
    model = MaskablePPO.load(model_path)
    
    print(f"Creating evaluation environment...")
    env = CuttleRLEnvironment()
    
    # Statistics
    wins = 0
    losses = 0
    stalemates = 0
    total_rewards = []
    episode_lengths = []
    invalid_actions = 0
    
    print(f"Running {n_episodes} evaluation episodes with action masking...")
    for episode in range(n_episodes):
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}")
        
        episode_reward, steps, winner = play_episode(model, env, deterministic=True)
        
        # Record results
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Categorize outcome
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            stalemates += 1
    
    # Print results
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({n_episodes} episodes)")
    print("=" * 50)
    print(f"Win Rate:       {wins/n_episodes*100:6.1f}%  ({wins} wins)")
    print(f"Loss Rate:      {losses/n_episodes*100:6.1f}%  ({losses} losses)")
    print(f"Stalemate Rate: {stalemates/n_episodes*100:6.1f}%  ({stalemates} stalemates)")
    print("-" * 50)
    print(f"Average Reward:        {np.mean(total_rewards):7.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):6.1f} steps")
    print("=" * 50)


def main():
    """Main evaluation function."""
    model_path = os.path.join(MODEL_DIR, "cuttle_rl_final")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}.zip")
        print("Please train a model first using: make train-rl")
        return
    
    evaluate_agent(model_path, n_episodes=100)


if __name__ == "__main__":
    main()
