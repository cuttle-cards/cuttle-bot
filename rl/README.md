# RL Training for Cuttle Game

Reinforcement Learning training for the Cuttle card game using **MaskablePPO** (Proximal Policy Optimization with action masking) from Stable Baselines3.

## Quick Start

### Train a Model

```bash
# Full training (500K timesteps, ~2-3 hours)
make train-rl

# Quick test (10K timesteps, ~2-3 minutes)
make test-rl
```

### Evaluate a Trained Model

```bash
make eval-rl
```

### Monitor Training

```bash
make tensorboard
# Open http://localhost:6006
```

## Hyperparameter Search

Test multiple configurations to find the best settings:

```bash
# Quick search (50K steps each, ~1 hour total)
make hypersearch-quick-rl

# Full search (200K steps each, ~3-4 hours)
make hypersearch-rl

# Compare results
make compare-rl DIR=rl/experiments/<timestamp>
```

## Debugging Tools

```bash
# Generate detailed gameplay logs
make debug-rl

# Analyze action patterns
make analyze-rl

# View individual games interactively
make view-rl
```

---

## Architecture

### Files

| File                       | Purpose                                   |
| -------------------------- | ----------------------------------------- |
| `config.py`                | Hyperparameters and reward settings       |
| `cuttle_env.py`            | Gymnasium environment with action masking |
| `self_play_env.py`         | Self-play wrapper for training            |
| `train.py`                 | Main training script                      |
| `evaluate.py`              | Model evaluation                          |
| `hyperparameter_search.py` | Automated config testing                  |
| `compare_experiments.py`   | Result analysis                           |
| `debug_gameplay.py`        | Generate debug logs                       |
| `analyze_logs.py`          | Pattern analysis                          |
| `view_game.py`             | Interactive game viewer                   |
| `game_logger.py`           | Logging implementation                    |

### Environment

- **Observation space**: 610-dimensional vector encoding game state (hand, fields, scores, flags, discard, revealed)
- **Action space**: Discrete(8478) with fixed card-identity mapping (not per-turn legal-action indices)
- **Action masking**: Only legal actions are considered by the policy

---

## Action Space Mapping (Fixed Indices)

Action indices are stable across turns and map to a specific semantic action based on **card identity** (rank + suit), not on a per-turn legal-action list. The mapping lives in `rl/action_mapping.py`.

### Card Identity Index

Each card maps to `0..51` using a canonical order:

```
card_index = (rank_value - 1) * 4 + suit_value
```

- `rank_value` comes from `game/card.py` `Rank` enum (Ace=1 .. King=13).
- `suit_value` comes from `Suit` enum (Clubs=0 .. Spades=3).

### Action Groups and Offsets

Action indices are grouped; each group has a fixed size and offset:

1. Draw: `1`
2. Resolve one-off: `1`
3. Play points (card identity): `52`
4. Play face card (card identity): `52`
5. Play one-off (untargeted, card identity): `52`
6. Play one-off (targeted: attacker, target): `52 * 52`
7. Counter (two) (card identity): `52`
8. Take from discard (card identity): `52`
9. Discard from hand (four) (card identity): `52`
10. Discard revealed (seven) (card identity): `52`
11. Scuttle (attacker, target): `52 * 52`
12. Jack (attacker, target): `52 * 52`

Total size: `8478`.

For paired actions (scuttle/jack/targeted one-off), the pair index is:

```
pair_index = attacker_index * 52 + target_index
```

### Mapping to Concrete Actions

The environment builds a mapping each step using `game_state.get_legal_actions()`:

1. `build_action_map(legal_actions)` converts each `Action` into a fixed index.
2. `action_masks()` marks only those indices as legal.
3. `step(action_index)` resolves the index back to the matching `Action`.

### Illegal Action Handling

If a predicted index does **not** map to a legal action:

- The environment returns `invalid_action_penalty` and ends the episode early.
- This should not occur when action masking is properly applied.

### Index Calculation Examples

Assume `Rank` and `Suit` enum values from `game/card.py` (Clubs=0, Diamonds=1, Hearts=2, Spades=3) and:

```
card_index = (rank_value - 1) * 4 + suit_value
```

Also assume the group offsets from `rl/action_mapping.py`:

```
draw=0
resolve=1
points=2
face=54
one_off=106
one_off_target=158
counter=2862
take_from_discard=2914
discard_from_hand=2966
discard_revealed=3018
scuttle=3070
jack=5774
```

Examples:

- **Play points**: `10 of Hearts` (rank=10, suit=Hearts=2)  
  `card_index = (10-1)*4 + 2 = 38`  
  `action_index = points_offset + card_index = 2 + 38 = 40`

- **Seven one-off** (untargeted): `7 of Clubs` (rank=7, suit=Clubs=0)  
  `card_index = (7-1)*4 + 0 = 24`  
  `action_index = one_off_offset + card_index = 106 + 24 = 130`

- **Seven from revealed pile** (discard revealed): `7 of Clubs`  
  `action_index = discard_revealed_offset + card_index = 3018 + 24 = 3042`

---

## Key Findings (might be outdated)

### Best Configuration: Baseline (Minimal Reward Shaping)

After extensive hyperparameter search, the **simplest configuration** performed best:

```python
REWARD_CONFIG = {
    "win": 100.0,
    "loss": -100.0,
    "progress_multiplier": 0.1,   # Minimal
    "turn_penalty": -0.01,        # Minimal
}
```

**Why it works:**
- Sparse rewards (win/loss) let the agent learn actual game strategy
- Heavy reward shaping causes overfitting to intermediate rewards
- Agent learns to play the game, not exploit reward hacking

### Hyperparameter Search Results

| Config             | Eval Reward | Notes                               |
| ------------------ | ----------- | ----------------------------------- |
| **baseline**       | -4.31       | ✅ Best - won games, longer episodes |
| aggressive_scoring | -9.07       | ❌ Crashed early                     |
| high_progress      | -9.59       | ❌ Overfitted to progress            |
| fast_learning      | -9.86       | ❌ Unstable                          |
| conservative       | -9.97       | ❌ Too slow                          |

---

## Important: Action Masking

### The Bug We Fixed

MaskablePPO requires proper environment wrapping to use action masks:

```python
# WRONG - action masking doesn't work
env = CuttleRLEnvironment()
env = Monitor(env, LOG_DIR)

# CORRECT - action masking works
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    while hasattr(env, 'env'):
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        env = env.env
    return env.action_masks()

env = CuttleRLEnvironment()
env = SelfPlayWrapper(env)
env = Monitor(env, LOG_DIR)
env = ActionMasker(env, mask_fn)  # CRITICAL
```

### Why Action Masking Matters

Without it:
- Agent attempts invalid actions (-10 penalty each)
- Episodes crash after 1-2 steps
- No learning occurs

With it:
- Agent only sees legal actions
- Full games play out (50-150 steps)
- Agent learns actual strategy

---

## Configuration Reference

### Training Parameters

```python
TRAINING_CONFIG = {
    "total_timesteps": 500000,  # How long to train
    "learning_rate": 3e-4,      # Adam optimizer LR
    "n_steps": 2048,            # Steps before policy update
    "batch_size": 64,           # Minibatch size
    "n_epochs": 10,             # Epochs per update
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE parameter
    "clip_range": 0.2,          # PPO clip range
    "ent_coef": 0.01,           # Entropy coefficient
}
```

### Reward Parameters

```python
REWARD_CONFIG = {
    "win": 100.0,               # Terminal reward for winning
    "loss": -100.0,             # Terminal penalty for losing
    "stalemate": -50.0,         # Penalty for draw
    "progress_multiplier": 0.1, # Intermediate reward multiplier
    "turn_penalty": -0.01,      # Per-turn penalty
    "invalid_action_penalty": -10.0,  # Safety check
}
```

### Parameter Guidelines

| Parameter             | Too Low       | Recommended   | Too High    |
| --------------------- | ------------- | ------------- | ----------- |
| `learning_rate`       | Slow learning | 1e-4 to 3e-4  | Unstable    |
| `progress_multiplier` | No guidance   | 0.1 to 1.0    | Overfitting |
| `turn_penalty`        | Long games    | -0.01 to -0.1 | Rushed play |

---

## Troubleshooting

### High Timeout Rate

**Symptoms:** Games exceed 200 steps frequently

**Solutions:**
1. Increase `turn_penalty` slightly
2. Check for resolve loops in gameplay logs
3. Verify action masking is working

### Negative Eval Rewards

**Symptoms:** Agent loses more than wins

**Solutions:**
1. Train longer (500K+ timesteps)
2. Reduce reward shaping (use baseline config)
3. Check for invalid action penalties

### Short Eval Episodes

**Symptoms:** Episodes only 1-5 steps

**Cause:** Action masking not working, or overfitted model

**Solution:** Verify `ActionMasker` wrapper is applied correctly

### Training Instability

**Symptoms:** Reward oscillates wildly, NaN losses

**Solutions:**
1. Reduce `learning_rate` by 10x
2. Increase `batch_size`
3. Check reward calculations for bugs

---

## Development Notes

### Testing Changes

```bash
# Run quick training test
make test-rl

# Generate debug logs
make debug-rl

# Analyze patterns
make analyze-rl
```

### Viewing Results

```bash
# TensorBoard
tensorboard --logdir rl/logs

# Compare experiments
make compare-rl DIR=rl/experiments/<timestamp>

# Interactive game viewer
make view-rl
```

### Output Locations

| Output         | Location            |
| -------------- | ------------------- |
| Trained models | `rl/models/`        |
| Training logs  | `rl/logs/`          |
| Experiments    | `rl/experiments/`   |
| Gameplay logs  | `rl/gameplay_logs/` |

---

## Changelog

### 2026-01-25: Action Masking Fix

**Problem:** MaskablePPO wasn't receiving action masks correctly.

**Impact:** 
- Agents attempted invalid actions constantly
- Episodes crashed after 1-2 steps
- All previous training was invalid

**Solution:** Added `ActionMasker` wrapper with proper environment unwrapping.

**Files changed:**
- `train.py` - Added ActionMasker
- `hyperparameter_search.py` - Added ActionMasker
- `debug_gameplay.py` - Added ActionMasker

### 2026-01-25: Hyperparameter Search

**Finding:** Baseline config (minimal reward shaping) outperforms all others.

**Implication:** Sparse rewards work better than dense reward shaping for this game.

**Config adopted:**
```python
progress_multiplier: 0.1  # Was 10.0
turn_penalty: -0.01       # Was -1.0
```

### 2026-01-25: Debug Tools

**Added:**
- `game_logger.py` - Step-by-step game logging
- `analyze_logs.py` - Automated pattern analysis
- `view_game.py` - Interactive game viewer
- `compare_experiments.py` - Experiment comparison

**Commands:**
- `make debug-rl`
- `make analyze-rl`
- `make view-rl`
- `make compare-rl`

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
