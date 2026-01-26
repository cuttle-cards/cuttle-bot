"""Analyze RL gameplay logs to identify issues."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def analyze_logs(log_dir: str = "rl/gameplay_logs") -> None:
    """Analyze gameplay logs to identify patterns and issues.
    
    Args:
        log_dir: Directory containing log files
    """
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"❌ No logs found at {log_dir}")
        print("   Run 'make debug-rl' first to generate logs")
        return
    
    # Find all game logs
    game_files = sorted(log_path.glob("game_*.json"))
    
    if not game_files:
        print(f"❌ No game logs found in {log_dir}")
        return
    
    print(f"\n{'='*70}")
    print("RL GAMEPLAY ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Analyzing {len(game_files)} games...\n")
    
    # Collect statistics
    action_types = Counter()
    action_patterns = defaultdict(list)
    timeout_games = []
    quick_wins = []
    
    for game_file in game_files:
        with open(game_file, "r") as f:
            game_data = json.load(f)
        
        game_id = game_data["game_id"]
        steps = game_data["steps"]
        outcome = game_data["outcome"]
        
        # Count action types
        for step in steps:
            action_type = step["action"]["type"]
            action_types[action_type] += 1
        
        # Detect patterns
        recent_actions = [s["action"]["type"] for s in steps[-20:]]
        pattern_key = " -> ".join(recent_actions[-5:]) if len(recent_actions) >= 5 else ""
        action_patterns[pattern_key].append(game_id)
        
        # Categorize games
        if outcome["reason"] == "timeout":
            timeout_games.append({
                "id": game_id,
                "steps": outcome["total_steps"],
                "final_scores": outcome["final_scores"],
                "recent_actions": recent_actions[-10:],
            })
        elif outcome["total_steps"] < 50 and outcome["reason"] == "win":
            quick_wins.append({
                "id": game_id,
                "steps": outcome["total_steps"],
                "winner": outcome["winner"],
            })
    
    # Print analysis
    print("📊 ACTION TYPE DISTRIBUTION")
    print("-" * 70)
    total_actions = sum(action_types.values())
    for action_type, count in action_types.most_common():
        percentage = (count / total_actions) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {action_type:20s} {count:5d} ({percentage:5.1f}%) {bar}")
    
    print(f"\n🔄 TIMEOUT GAMES: {len(timeout_games)}/{len(game_files)}")
    print("-" * 70)
    if timeout_games:
        for game in timeout_games[:5]:  # Show first 5
            print(f"\n  Game {game['id']}:")
            print(f"    Steps: {game['steps']}")
            print(f"    Final scores: P0={game['final_scores']['player_0']}, "
                  f"P1={game['final_scores']['player_1']}")
            print(f"    Last 10 actions: {' -> '.join(game['recent_actions'])}")
        
        if len(timeout_games) > 5:
            print(f"\n  ... and {len(timeout_games) - 5} more timeout games")
    
    print(f"\n⚡ QUICK WINS: {len(quick_wins)}/{len(game_files)}")
    print("-" * 70)
    if quick_wins:
        for game in quick_wins:
            print(f"  Game {game['id']}: Winner P{game['winner']} in {game['steps']} steps")
    
    # Detect stuck patterns
    print(f"\n🔍 COMMON ACTION PATTERNS (last 5 moves)")
    print("-" * 70)
    common_patterns = [(p, len(games)) for p, games in action_patterns.items() 
                      if len(games) > 1 and p]
    common_patterns.sort(key=lambda x: x[1], reverse=True)
    
    for pattern, count in common_patterns[:10]:
        print(f"  [{count} games] {pattern}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 70)
    
    draw_percentage = (action_types.get("Draw", 0) / total_actions) * 100
    if draw_percentage > 40:
        print("  ⚠️  HIGH DRAW RATE: Bot is drawing too often without playing cards")
        print("     Consider adjusting reward to penalize excessive draws")
    
    if len(timeout_games) / len(game_files) > 0.5:
        print("  ⚠️  HIGH TIMEOUT RATE: Games are not progressing")
        print("     Bot may not understand how to play for points")
        print("     Consider:")
        print("       - Increase reward for playing point cards")
        print("       - Add reward shaping for field control")
        print("       - Reduce max_steps or add progress penalty")
    
    points_percentage = (action_types.get("Points", 0) / total_actions) * 100
    if points_percentage < 10:
        print("  ⚠️  LOW POINTS PLAY: Bot rarely plays point cards")
        print("     Increase reward for point cards significantly")
    
    print(f"\n{'='*70}\n")
    print(f"📁 Full logs available at: {log_path.absolute()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    analyze_logs()
