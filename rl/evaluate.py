"""Evaluate trained RL agent with action masking."""
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sb3_contrib import MaskablePPO

from rl.action_mapping import action_index_to_action
from rl.config import LOG_DIR, MODEL_DIR
from rl.cuttle_env import CuttleRLEnvironment


def _snapshot_game_state(env: CuttleRLEnvironment) -> Dict[str, Any]:
    """Capture a compact, readable snapshot of the current game state."""
    if not env.game:
        return {}
    state = env.game.game_state
    return {
        "turn": state.turn,
        "current_action_player": state.current_action_player,
        "overall_turn": state.overall_turn,
        "scores": {
            "player_0": state.get_player_score(0),
            "player_1": state.get_player_score(1),
        },
        "targets": {
            "player_0": state.get_player_target(0),
            "player_1": state.get_player_target(1),
        },
        "hands": [
            [str(card) for card in state.hands[0]],
            [str(card) for card in state.hands[1]],
        ],
        "fields": [
            [str(card) for card in state.get_player_field(0)],
            [str(card) for card in state.get_player_field(1)],
        ],
        "deck_count": len(state.deck),
        "discard_pile": [str(card) for card in state.discard_pile],
        "resolving_one_off": state.resolving_one_off,
        "resolving_three": state.resolving_three,
        "resolving_seven": state.resolving_seven,
        "pending_three_player": state.pending_three_player,
        "pending_four_player": state.pending_four_player,
        "pending_four_count": state.pending_four_count,
        "pending_seven_requires_discard": state.pending_seven_requires_discard,
    }


def play_episode(
    model: MaskablePPO, 
    env: CuttleRLEnvironment, 
    deterministic: bool = True,
    record: bool = False,
) -> Tuple[float, int, Optional[int], Optional[Dict[str, Any]]]:
    """Play one episode with action masking."""
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    trace: Optional[Dict[str, Any]] = {"steps": []} if record else None
    
    while not done:
        # Agent's turn with action mask
        action_mask = env.action_masks()
        obs_before = obs
        action, _ = model.predict(
            obs,
            action_masks=action_mask,  # Pass mask to model
            deterministic=deterministic
        )
        if env.game:
            action_obj = action_index_to_action(env.game.game_state, int(action))
            legal_actions = env.game.game_state.get_legal_actions()
        else:
            action_obj = None
            legal_actions = []
        state_before = _snapshot_game_state(env) if record else None

        obs, reward, done, truncated, info = env.step(action)
        state_after = _snapshot_game_state(env) if record else None
        episode_reward += reward
        steps += 1

        if trace is not None:
            trace["steps"].append(
                {
                    "actor": "agent",
                    "step": steps,
                    "obs": obs_before.tolist(),
                    "next_obs": obs.tolist(),
                    "action_index": int(action),
                    "action": str(action_obj) if action_obj else None,
                    "legal_actions": [str(a) for a in legal_actions],
                    "action_mask": action_mask.astype(int).tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "info": info,
                    "state_before": state_before,
                    "state_after": state_after,
                }
            )
        
        if done:
            break
        
        # Random opponent's turn (also uses masking)
        opponent_mask = env.action_masks()
        legal_indices = np.where(opponent_mask)[0]
        if len(legal_indices) > 0:
            opp_action = np.random.choice(legal_indices)
            obs_before = obs
            if env.game:
                opp_action_obj = action_index_to_action(env.game.game_state, int(opp_action))
                opp_legal_actions = env.game.game_state.get_legal_actions()
            else:
                opp_action_obj = None
                opp_legal_actions = []
            state_before = _snapshot_game_state(env) if record else None

            obs, opp_reward, done, truncated, info = env.step(opp_action)
            state_after = _snapshot_game_state(env) if record else None
            episode_reward -= opp_reward
            steps += 1

            if trace is not None:
                trace["steps"].append(
                    {
                        "actor": "opponent",
                        "step": steps,
                        "obs": obs_before.tolist(),
                        "next_obs": obs.tolist(),
                        "action_index": int(opp_action),
                        "action": str(opp_action_obj) if opp_action_obj else None,
                        "legal_actions": [str(a) for a in opp_legal_actions],
                        "action_mask": opponent_mask.astype(int).tolist(),
                        "reward": float(opp_reward),
                        "done": bool(done),
                        "truncated": bool(truncated),
                        "info": info,
                        "state_before": state_before,
                        "state_after": state_after,
                    }
                )
    
    # Get winner
    winner = env.game.game_state.winner() if env.game else None
    
    if trace is not None:
        trace["summary"] = {
            "episode_reward": float(episode_reward),
            "steps": steps,
            "winner": winner,
            "deterministic": deterministic,
        }

    return episode_reward, steps, winner, trace


def evaluate_agent(
    model_path: str,
    n_episodes: int = 100,
    record_path: Optional[str] = None,
):
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
        
        record = record_path is not None and episode == 0
        episode_reward, steps, winner, trace = play_episode(
            model, env, deterministic=True, record=record
        )
        
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

        if record and trace is not None:
            os.makedirs(os.path.dirname(record_path), exist_ok=True)
            with open(record_path, "w", encoding="utf-8") as handle:
                json.dump(trace, handle, indent=2)
            print(f"Saved episode trace to: {record_path}")
    
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
    record_path = os.path.join(LOG_DIR, "eval_rollout.json")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}.zip")
        print("Please train a model first using: make train-rl")
        return
    
    evaluate_agent(model_path, n_episodes=100, record_path=record_path)


if __name__ == "__main__":
    main()
