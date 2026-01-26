"""Debug script to analyze RL gameplay with detailed logging."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from rl.cuttle_env import CuttleRLEnvironment
from rl.self_play_env import SelfPlayWrapper


def mask_fn(env):
    """Function that returns action mask for MaskablePPO."""
    # Unwrap to get to the actual environment with action_masks method
    while hasattr(env, 'env'):
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        env = env.env
    return env.action_masks()


def run_debug_games(num_games: int = 10, model_path: str = "rl/models/best_model.zip"):
    """Run games with detailed logging for debugging.
    
    Args:
        num_games: Number of games to play and log
        model_path: Path to trained model (or None to use random actions)
    """
    print(f"\n{'='*60}")
    print("DEBUGGING RL GAMEPLAY")
    print(f"{'='*60}\n")
    
    # Create environment with logging enabled
    base_env = CuttleRLEnvironment(enable_logging=True)
    env = SelfPlayWrapper(base_env)
    env = ActionMasker(env, mask_fn)  # Critical: wrap with ActionMasker
    
    # Load model if available
    try:
        model = MaskablePPO.load(model_path)
        print(f"✅ Loaded model from: {model_path}\n")
        use_model = True
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("Using random actions instead\n")
        use_model = False
    
    # Run games
    for game_num in range(num_games):
        print(f"Playing game {game_num + 1}/{num_games}...", end=" ")
        
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            action_masks = env.action_masks()
            
            if use_model:
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
            else:
                # Random action from legal actions
                import numpy as np
                legal_actions = np.where(action_masks)[0]
                action = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
            
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            if done or truncated:
                break
        
        print(f"Finished in {step_count} steps")
    
    # Generate summary
    if base_env.logger:
        base_env.logger.generate_summary()
    
    print("\n💡 TIP: Check the JSON logs in rl/gameplay_logs/ for detailed analysis")
    print("   Each file contains step-by-step actions, game state, and outcomes\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug RL gameplay with detailed logs")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--model", type=str, default="rl/models/best_model.zip", 
                       help="Path to model")
    
    args = parser.parse_args()
    
    run_debug_games(num_games=args.games, model_path=args.model)
