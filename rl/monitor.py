#!/usr/bin/env python3
"""Simple training monitor for RL training."""
import re
import sys
import time
import os
import glob

MONITOR_CSV = "/Users/lihao/Documents/Projects/cuttle-bot/rl/logs/monitor.csv"

def find_latest_log() -> str:
    """Find the most recent training log file."""
    # Prefer explicit current log if present
    current_log = "/tmp/train_current.log"
    if os.path.exists(current_log) and os.path.getsize(current_log) > 100:
        return current_log

    # Use glob to find all training logs
    all_logs = list(set(glob.glob("/tmp/train*.log")))

    # Filter to logs with content and return most recently modified
    valid = [
        (path, os.path.getmtime(path))
        for path in all_logs
        if os.path.exists(path) and os.path.getsize(path) > 100
    ]
    if valid:
        return max(valid, key=lambda entry: entry[1])[0]

    # Fallback
    return current_log


def _read_monitor_episode_stats() -> dict:
    """Read episode count and average length from Monitor CSV."""
    if not os.path.exists(MONITOR_CSV):
        return {"episodes": 0, "avg_len": 0.0}

    with open(MONITOR_CSV, "r") as monitor_file:
        lines = [line.strip() for line in monitor_file if line.strip()]

    if not lines:
        return {"episodes": 0, "avg_len": 0.0}

    # Find last header line to avoid mixing runs
    last_header_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            last_header_idx = idx

    data_lines = lines[last_header_idx + 2 :]  # skip header + column line
    lengths = []
    for line in data_lines:
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                lengths.append(float(parts[1]))
            except ValueError:
                continue

    if not lengths:
        return {"episodes": 0, "avg_len": 0.0}

    return {"episodes": len(lengths), "avg_len": sum(lengths) / len(lengths)}

def monitor(log_file: str | None = None, refresh: bool = False) -> None:
    """Monitor training progress."""
    if log_file is None:
        log_file = find_latest_log()
    
    while True:
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return
            
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Parse metrics
        timesteps = re.findall(r'total_timesteps\s+\|\s+(\d+)', content)
        ep_rew = re.findall(r'ep_rew_mean\s+\|\s+([-\d.]+)', content)
        ep_len = re.findall(r'ep_len_mean\s+\|\s+([\d.]+)', content)
        fps_vals = re.findall(r'fps\s+\|\s+(\d+)', content)
        time_elapsed = re.findall(r'time_elapsed\s+\|\s+(\d+)', content)
        
        total = 500000
        
        # Clear screen if refreshing
        if refresh:
            print("\033[2J\033[H", end="")
        
        print("=" * 60)
        print("🎮 CUTTLE RL TRAINING MONITOR")
        print(f"   Log: {os.path.basename(log_file)}")
        print("=" * 60)
        
        if timesteps:
            latest = int(timesteps[-1])
            pct = (latest / total) * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            
            print(f"\n📊 Progress: {latest:,} / {total:,} ({pct:.1f}%)")
            print(f"   [{bar}]")
        
        if ep_rew:
            current = float(ep_rew[-1])
            print(f"\n📈 Reward: {current:.2f}", end="")
            if len(ep_rew) >= 5:
                early = sum(float(r) for r in ep_rew[:5]) / 5
                recent = sum(float(r) for r in ep_rew[-5:]) / 5
                change = recent - early
                print(f"  (trend: {'+' if change > 0 else ''}{change:.2f})")
            else:
                print()
        
        if ep_len:
            print(f"🎲 Episode Length: {float(ep_len[-1]):.1f} steps")
        
        if fps_vals and timesteps:
            fps = int(fps_vals[-1])
            remaining = total - int(timesteps[-1])
            eta_min = (remaining / fps) / 60 if fps > 0 else 0
            print(f"⏱️  Speed: {fps:,}/s | ETA: {eta_min:.1f} min")
        
        # Calculate timeout stats
        timeouts = content.count("TIMEOUT")
        stalls = content.count("STALL")
        invalid = content.count("Invalid action")
        episode_stats = _read_monitor_episode_stats()
        episodes = episode_stats["episodes"]

        if episodes > 0:
            timeout_pct = (timeouts / episodes * 100)
            stall_pct = (stalls / episodes * 100)
            non_finish_pct = ((timeouts + stalls) / episodes * 100)
            avg_len = episode_stats["avg_len"]
            print(f"\n⚠️  Timeouts: {timeouts} / {episodes} games ({timeout_pct:.1f}%)")
            if stalls > 0:
                print(f"⚠️  Stalls:   {stalls} / {episodes} games ({stall_pct:.1f}%)")
            print(f"⚠️  Non-finish rate: {non_finish_pct:.1f}%")
            print(f"   Avg episode length: {avg_len:.1f} steps")

            # Visual indicator
            if non_finish_pct > 90:
                print("   Status: 🔴 Most games end without a winner")
            elif non_finish_pct > 50:
                print("   Status: 🟡 Many games stall/timeout")
            elif non_finish_pct > 20:
                print("   Status: 🟢 Good progress (agent winning more)")
            else:
                print("   Status: ✅ Excellent (agent wins most games)")
        else:
            print(f"\n⚠️  Timeouts: {timeouts}")
        
        if invalid > 0:
            print(f"❌ Invalid actions: {invalid}")
        
        # Self-play info
        model_probs = re.findall(r'opponent model prob = ([\d.]+)%', content)
        if model_probs:
            current_prob = float(model_probs[-1])
            print(f"\n🤖 Self-Play: {current_prob:.0f}% model opponent")
        elif "Self-play initialized" in content:
            print(f"\n🤖 Self-Play: Starting (0% model, 100% random)")
        
        print("=" * 60)
        
        if not refresh:
            break
            
        time.sleep(5)


if __name__ == "__main__":
    refresh = "--watch" in sys.argv or "-w" in sys.argv
    monitor(refresh=refresh)
