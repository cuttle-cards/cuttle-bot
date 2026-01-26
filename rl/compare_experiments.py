"""Compare results from multiple experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_monitor_data(monitor_file: Path) -> Dict[str, Any]:
    """Load data from a stable-baselines3 monitor file.
    
    Args:
        monitor_file: Path to monitor.csv file
        
    Returns:
        Dictionary with episode rewards and lengths
    """
    if not monitor_file.exists():
        return {"rewards": [], "lengths": [], "times": []}
    
    rewards = []
    lengths = []
    times = []
    
    with open(monitor_file, "r") as f:
        # Skip header lines
        for _ in range(2):
            f.readline()
        
        # Read data
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        rewards.append(float(parts[0]))
                        lengths.append(int(parts[1]))
                        times.append(float(parts[2]))
                    except ValueError:
                        continue
    
    return {
        "rewards": rewards,
        "lengths": lengths,
        "times": times,
    }


def analyze_experiment(exp_path: Path) -> Dict[str, Any]:
    """Analyze results from a single experiment.
    
    Args:
        exp_path: Path to experiment directory
        
    Returns:
        Dictionary with analysis results
    """
    # Load config
    config_file = exp_path / "config.json"
    if not config_file.exists():
        return {"error": "No config.json found"}
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    # Load training monitor data
    train_monitor = exp_path / "train.monitor.csv"
    train_data = load_monitor_data(train_monitor)
    
    # Load evaluation monitor data  
    eval_monitor = exp_path / "eval.monitor.csv"
    eval_data = load_monitor_data(eval_monitor)
    
    # Calculate statistics
    analysis = {
        "name": config["name"],
        "config": config,
        "train": {
            "total_episodes": len(train_data["rewards"]),
            "mean_reward": float(np.mean(train_data["rewards"])) if train_data["rewards"] else 0.0,
            "std_reward": float(np.std(train_data["rewards"])) if train_data["rewards"] else 0.0,
            "mean_length": float(np.mean(train_data["lengths"])) if train_data["lengths"] else 0.0,
            "final_100_mean_reward": float(np.mean(train_data["rewards"][-100:])) if len(train_data["rewards"]) >= 100 else 0.0,
        },
        "eval": {
            "total_episodes": len(eval_data["rewards"]),
            "mean_reward": float(np.mean(eval_data["rewards"])) if eval_data["rewards"] else 0.0,
            "std_reward": float(np.std(eval_data["rewards"])) if eval_data["rewards"] else 0.0,
            "mean_length": float(np.mean(eval_data["lengths"])) if eval_data["lengths"] else 0.0,
            "best_reward": float(max(eval_data["rewards"])) if eval_data["rewards"] else 0.0,
        },
    }
    
    # Check for timeout issues
    timeout_rate = sum(1 for length in train_data["lengths"] if length >= 200) / max(len(train_data["lengths"]), 1)
    analysis["train"]["timeout_rate"] = float(timeout_rate)
    
    return analysis


def compare_experiments(experiments_dir: Path) -> None:
    """Compare all experiments in a directory.
    
    Args:
        experiments_dir: Directory containing experiment subdirectories
    """
    if not experiments_dir.exists():
        print(f"❌ Directory not found: {experiments_dir}")
        return
    
    # Find all experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    
    if not exp_dirs:
        print(f"❌ No experiments found in {experiments_dir}")
        return
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    print(f"Found {len(exp_dirs)} experiments\n")
    
    # Analyze all experiments
    analyses = []
    for exp_dir in sorted(exp_dirs):
        analysis = analyze_experiment(exp_dir)
        if "error" not in analysis:
            analyses.append(analysis)
    
    if not analyses:
        print("❌ No valid experiments to compare")
        return
    
    # Sort by evaluation mean reward (best first)
    analyses.sort(key=lambda x: x["eval"]["mean_reward"], reverse=True)
    
    # Print comparison table
    print(f"{'Rank':<6} {'Name':<25} {'Train Reward':<15} {'Eval Reward':<15} {'Timeout %':<12} {'Avg Length':<12}")
    print(f"{'-'*6} {'-'*25} {'-'*15} {'-'*15} {'-'*12} {'-'*12}")
    
    for i, analysis in enumerate(analyses, 1):
        train_reward = analysis["train"]["final_100_mean_reward"]
        eval_reward = analysis["eval"]["mean_reward"]
        timeout_pct = analysis["train"]["timeout_rate"] * 100
        avg_length = analysis["train"]["mean_length"]
        
        print(f"{i:<6} {analysis['name']:<25} {train_reward:>14.2f} {eval_reward:>14.2f} {timeout_pct:>11.1f} {avg_length:>11.1f}")
    
    # Detailed analysis of top 3
    print(f"\n{'='*80}")
    print("TOP 3 EXPERIMENTS (Detailed)")
    print(f"{'='*80}\n")
    
    for i, analysis in enumerate(analyses[:3], 1):
        print(f"{i}. {analysis['name']}")
        print(f"   {'-'*76}")
        print(f"   Description: {analysis['config']['description']}")
        print(f"\n   Training Config:")
        for key, value in analysis['config']['training'].items():
            print(f"     {key:20s}: {value}")
        print(f"\n   Reward Config:")
        for key, value in analysis['config']['reward'].items():
            print(f"     {key:20s}: {value}")
        print(f"\n   Training Results:")
        print(f"     Episodes:            {analysis['train']['total_episodes']}")
        print(f"     Mean Reward:         {analysis['train']['mean_reward']:.2f} ± {analysis['train']['std_reward']:.2f}")
        print(f"     Final 100 Reward:    {analysis['train']['final_100_mean_reward']:.2f}")
        print(f"     Mean Episode Length: {analysis['train']['mean_length']:.1f}")
        print(f"     Timeout Rate:        {analysis['train']['timeout_rate']*100:.1f}%")
        print(f"\n   Evaluation Results:")
        print(f"     Episodes:            {analysis['eval']['total_episodes']}")
        print(f"     Mean Reward:         {analysis['eval']['mean_reward']:.2f} ± {analysis['eval']['std_reward']:.2f}")
        print(f"     Best Reward:         {analysis['eval']['best_reward']:.2f}")
        print(f"     Mean Episode Length: {analysis['eval']['mean_length']:.1f}")
        print()
    
    # Recommendations
    print(f"{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    best = analyses[0]
    
    print(f"🏆 Best performing: {best['name']}")
    print(f"   Mean eval reward: {best['eval']['mean_reward']:.2f}")
    
    if best['train']['timeout_rate'] > 0.3:
        print(f"\n⚠️  Warning: High timeout rate ({best['train']['timeout_rate']*100:.1f}%)")
        print("   Consider:")
        print("     - Increase progress_multiplier")
        print("     - Increase turn_penalty")
        print("     - Train for more timesteps")
    
    if best['eval']['mean_reward'] < 0:
        print(f"\n⚠️  Warning: Negative mean reward")
        print("   The agent is losing more than winning. Consider:")
        print("     - Adjusting reward shaping")
        print("     - Training for more timesteps")
        print("     - Using a different learning rate")
    
    print(f"\n📊 View detailed metrics:")
    print(f"   tensorboard --logdir {experiments_dir.absolute()}")
    
    print(f"\n🎮 Test best model:")
    best_model = experiments_dir / best['name'] / "best_model" / "best_model.zip"
    if best_model.exists():
        print(f"   make debug-rl MODEL={best_model}")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "experiments_dir",
        type=Path,
        help="Directory containing experiments",
    )
    
    args = parser.parse_args()
    
    compare_experiments(args.experiments_dir)
