"""Hyperparameter search for RL training."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

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


# Define hyperparameter configurations to test
EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "baseline",
        "description": "Current baseline configuration",
        "training": {
            "total_timesteps": 200_000,
            "n_steps": 2048,
            "batch_size": 64,
            "learning_rate": 3e-4,
        },
        "reward": {
            "win": 100.0,
            "loss": -100.0,
            "stalemate": -50.0,
            "invalid_action_penalty": -10.0,
            "progress_multiplier": 0.1,
            "turn_penalty": -0.01,
        },
    },
    {
        "name": "high_progress_reward",
        "description": "Emphasize progress toward winning",
        "training": {
            "total_timesteps": 200_000,
            "n_steps": 2048,
            "batch_size": 64,
            "learning_rate": 3e-4,
        },
        "reward": {
            "win": 100.0,
            "loss": -100.0,
            "stalemate": -50.0,
            "invalid_action_penalty": -10.0,
            "progress_multiplier": 10.0,  # 100x increase
            "turn_penalty": -0.5,          # Penalize longer games
        },
    },
    {
        "name": "fast_learning",
        "description": "Higher learning rate for faster initial learning",
        "training": {
            "total_timesteps": 200_000,
            "n_steps": 1024,              # Smaller steps
            "batch_size": 128,            # Larger batches
            "learning_rate": 1e-3,        # Higher LR
        },
        "reward": {
            "win": 100.0,
            "loss": -100.0,
            "stalemate": -50.0,
            "invalid_action_penalty": -10.0,
            "progress_multiplier": 5.0,
            "turn_penalty": -0.2,
        },
    },
    {
        "name": "conservative",
        "description": "Lower LR, larger batches for stable learning",
        "training": {
            "total_timesteps": 200_000,
            "n_steps": 4096,              # Larger steps
            "batch_size": 32,             # Smaller batches
            "learning_rate": 1e-4,        # Lower LR
        },
        "reward": {
            "win": 100.0,
            "loss": -100.0,
            "stalemate": -50.0,
            "invalid_action_penalty": -10.0,
            "progress_multiplier": 3.0,
            "turn_penalty": -0.1,
        },
    },
    {
        "name": "aggressive_scoring",
        "description": "Heavy emphasis on scoring points",
        "training": {
            "total_timesteps": 200_000,
            "n_steps": 2048,
            "batch_size": 64,
            "learning_rate": 3e-4,
        },
        "reward": {
            "win": 100.0,
            "loss": -100.0,
            "stalemate": -50.0,
            "invalid_action_penalty": -10.0,
            "progress_multiplier": 20.0,  # Very high
            "turn_penalty": -1.0,         # Strong penalty for long games
        },
    },
]


def run_experiment(config: Dict[str, Any], experiment_dir: Path) -> Dict[str, Any]:
    """Run a single experiment with given configuration.
    
    Args:
        config: Experiment configuration
        experiment_dir: Directory to save experiment results
        
    Returns:
        Dictionary with experiment results
    """
    exp_name = config["name"]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")
    
    # Create experiment directory
    exp_path = experiment_dir / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Apply reward config (monkey patch for this experiment)
    import rl.config as rl_config
    for key, value in config["reward"].items():
        rl_config.REWARD_CONFIG[key] = value
    
    # Create environments with action masking
    train_env = SelfPlayWrapper(CuttleRLEnvironment())
    train_env = Monitor(train_env, str(exp_path / "train"))
    train_env = ActionMasker(train_env, mask_fn)  # Critical: wrap with ActionMasker
    
    eval_env = SelfPlayWrapper(CuttleRLEnvironment())
    eval_env = Monitor(eval_env, str(exp_path / "eval"))
    eval_env = ActionMasker(eval_env, mask_fn)  # Critical: wrap with ActionMasker
    
    # Training parameters
    training_config = config["training"]
    
    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        n_steps=training_config["n_steps"],
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        verbose=1,
        tensorboard_log=str(exp_path / "tensorboard"),
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(exp_path / "checkpoints"),
        name_prefix=f"{exp_name}_model",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(exp_path / "best_model"),
        log_path=str(exp_path / "eval_logs"),
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    # Train
    start_time = datetime.now()
    print(f"Training started at {start_time.isoformat()}\n")
    
    model.learn(
        total_timesteps=training_config["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nTraining completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Save final model
    model.save(exp_path / "final_model")
    
    # Collect results
    results = {
        "name": exp_name,
        "config": config,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "model_path": str(exp_path / "final_model.zip"),
        "best_model_path": str(exp_path / "best_model" / "best_model.zip"),
    }
    
    # Save results
    with open(exp_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return results


def run_all_experiments(
    configs: List[Dict[str, Any]],
    base_dir: str = "rl/experiments",
) -> None:
    """Run all experiments and save results.
    
    Args:
        configs: List of experiment configurations
        base_dir: Base directory for all experiments
    """
    experiment_dir = Path(base_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH")
    print(f"{'='*70}")
    print(f"Running {len(configs)} experiments")
    print(f"Results will be saved to: {experiment_dir.absolute()}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for i, config in enumerate(configs):
        print(f"\nExperiment {i+1}/{len(configs)}")
        try:
            results = run_experiment(config, experiment_dir)
            all_results.append(results)
        except Exception as e:
            print(f"❌ Experiment {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(configs),
        "successful_experiments": len(all_results),
        "experiments": all_results,
    }
    
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {experiment_dir.absolute()}")
    print(f"Successful: {len(all_results)}/{len(configs)}")
    print(f"{'='*70}\n")
    
    print("Next steps:")
    print(f"  1. Compare results: python rl/compare_experiments.py {experiment_dir}")
    print(f"  2. View tensorboard: tensorboard --logdir {experiment_dir}")
    print(f"  3. Test best model: make debug-rl --model <path_to_best>\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter search")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick experiments (50K timesteps each)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Run only specific configs by name",
    )
    
    args = parser.parse_args()
    
    # Filter configs if specified
    configs = EXPERIMENT_CONFIGS
    if args.configs:
        configs = [c for c in configs if c["name"] in args.configs]
        if not configs:
            print(f"❌ No configs found matching: {args.configs}")
            print(f"Available: {[c['name'] for c in EXPERIMENT_CONFIGS]}")
            sys.exit(1)
    
    # Reduce timesteps for quick mode
    if args.quick:
        print("🚀 Quick mode: Using 50K timesteps per experiment\n")
        for config in configs:
            config["training"]["total_timesteps"] = 50_000
    
    run_all_experiments(configs)
