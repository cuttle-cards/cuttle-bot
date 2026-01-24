# RL Training for Cuttle Game

Reinforcement Learning training setup for the Cuttle card game using **MaskablePPO** (Proximal Policy Optimization with action masking) from Stable Baselines3.

## Quick Start

### Train a Model

```bash
# Full training (100K timesteps, ~1-2 minutes)
make train-rl

# Or directly:
source cuttle-bot-3.12/bin/activate
PYTHONPATH=. python rl/train.py
```

### Evaluate a Trained Model

```bash
make eval-rl

# Or directly:
source cuttle-bot-3.12/bin/activate
PYTHONPATH=. python rl/evaluate.py
```

### Monitor Training

```bash
make tensorboard
# Then open http://localhost:6006
```

### Quick Test

```bash
# Quick test with 10K timesteps (~2-3 minutes)
make test-rl
```

## File Structure

```
rl/
├── README.md              # This file
├── config.py              # Hyperparameters and configuration
├── cuttle_env.py          # Gymnasium environment wrapper
├── self_play_env.py       # Self-play wrapper
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── models/                # Saved model checkpoints (gitignored)
│   └── cuttle_rl_final.zip
└── logs/                  # TensorBoard logs (gitignored)
```

## Key Features

- **Action Masking**: Agent only considers legal moves (no invalid action penalties)
- **State Encoding**: 206-dimensional observation vector encoding full game state
- **Self-Play**: Trains against random opponent (extensible to previous model checkpoints)
- **Checkpointing**: Auto-saves model every 10K timesteps
- **TensorBoard Logging**: Real-time training metrics visualization

## Configuration

All configuration is in `config.py`:

### Training Hyperparameters

```python
TRAINING_CONFIG = {
    "total_timesteps": 100000,  # Total training steps
    "learning_rate": 3e-4,      # Learning rate
    "n_steps": 2048,            # Steps per update
    "batch_size": 64,           # Minibatch size
    "n_epochs": 10,             # Epochs per update
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE parameter
    "clip_range": 0.2,          # PPO clip range
    "ent_coef": 0.01,           # Entropy coefficient
}
```

### Reward Structure

```python
REWARD_CONFIG = {
    "win": 100.0,                    # Win reward
    "loss": -100.0,                  # Loss penalty
    "stalemate": 0.0,                # Draw reward
    "progress_multiplier": 10.0,     # Score progress multiplier
    "turn_penalty": -1.0,            # Per-turn penalty
    "invalid_action_penalty": -50.0, # Shouldn't occur with masking
}
```

### Environment Config

```python
ENV_CONFIG = {
    "max_actions": 50,       # Max actions per turn
    "observation_dim": 206,  # State vector size
    "max_hand_size": 8,      # Max cards in hand
    "max_field_size": 10,    # Max cards on field
}
```

## How It Works

### Action Masking

The environment uses **action masking** to ensure the agent only considers legal moves:

1. `get_legal_actions()` returns list of valid `Action` objects
2. Action mask is boolean array: `True` for legal actions, `False` for illegal
3. Model predicts action index into legal actions list
4. Mask prevents model from selecting invalid actions

**Benefits**: Faster training, no wasted exploration on invalid moves.

### State Encoding

Game state is encoded as a **206-dimensional vector**:

- **Hand cards** (136 dims): 8 slots × 17 dims (suit + rank)
- **Opponent hand size** (1 dim): Normalized
- **Player 0 field** (30 dims): 10 slots × 3 dims
- **Player 1 field** (30 dims): 10 slots × 3 dims
- **Scores & targets** (4 dims): Normalized scores
- **Game flags** (5 dims): Current player, resolving flags, deck/discard sizes

### Training Flow

1. Environment resets to new game
2. Agent observes state (206-dim vector)
3. Agent predicts action (with masking)
4. Action executed, reward calculated
5. Opponent takes turn (random, also masked)
6. Repeat until game ends
7. Model updates using PPO algorithm

## Usage Examples

### Custom Training Run

```python
from rl import config
from rl.train import main

# Modify config
config.TRAINING_CONFIG["total_timesteps"] = 500000
config.TRAINING_CONFIG["learning_rate"] = 1e-4

# Train
main()
```

### Load and Use Model

```python
from sb3_contrib import MaskablePPO
from rl.cuttle_env import CuttleRLEnvironment

# Load model
model = MaskablePPO.load("rl/models/cuttle_rl_final")

# Create environment
env = CuttleRLEnvironment()
obs, info = env.reset()

# Get action with masking
action_mask = env.action_masks()
action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)

# Execute action
obs, reward, done, truncated, info = env.step(action)
```

### Evaluate Custom Model

```python
from rl.evaluate import evaluate_agent

# Evaluate specific model
evaluate_agent("rl/models/cuttle_rl_100000_steps", n_episodes=50)
```

## Output Files

### Models

- `rl/models/cuttle_rl_final.zip` - Final trained model
- `rl/models/cuttle_rl_10000_steps.zip` - Checkpoint at 10K steps
- `rl/models/cuttle_rl_20000_steps.zip` - Checkpoint at 20K steps
- etc.

### Logs

- `rl/logs/` - TensorBoard logs
  - View with: `make tensorboard`
  - Metrics: reward, episode length, policy loss, value loss, etc.

## Dependencies

Required packages (already in main `requirements.txt`):

```
gymnasium==0.29.1
stable-baselines3==2.2.1
sb3-contrib==2.2.1
torch>=2.0.0
tensorboard==2.15.1
numpy>=1.24.0
tqdm>=4.67.0
rich>=14.2.0
```

## Troubleshooting

### Model Not Found

```
ERROR: Model not found at rl/models/cuttle_rl_final.zip
```

**Solution**: Train a model first with `make train-rl`

### Import Errors

```
ImportError: You must install tqdm and rich...
```

**Solution**: Install missing packages:
```bash
source cuttle-bot-3.12/bin/activate
pip install tqdm rich
```

### Games Taking Too Long

Untrained agents may play very long games (both players just drawing cards). This is normal! The agent needs more training to learn strategic play.

**Solution**: 
- Train longer (increase `total_timesteps` in `config.py`)
- Adjust reward structure to encourage strategic moves
- Add episode length limits (see `cuttle_env.py`)

### Action Masking Not Working

If you see "WARNING: Invalid action attempted", action masking may not be properly passed to the model.

**Solution**: Ensure `action_masks` is passed to `model.predict()`:
```python
action_mask = env.action_masks()
action, _ = model.predict(obs, action_masks=action_mask)
```

## Performance

- **Training Speed**: ~1,200 FPS on modern CPU
- **100K Timesteps**: ~1-2 minutes
- **Model Size**: ~1-2 MB (compressed)
- **Memory Usage**: ~200-500 MB during training

## Key Concepts

### Action Masking

Action masking is **critical** for efficient training. Without it, the agent would waste time exploring invalid moves. The mask tells the model which actions are legal in the current state.

### Self-Play

Currently uses random opponent. Future improvements:
- Train against previous model checkpoints
- Use stronger opponents as agent improves
- Implement population-based training

### Reward Shaping

Rewards are designed to:
- Strongly reward winning (+100)
- Strongly penalize losing (-100)
- Provide intermediate feedback for score progress
- Slightly penalize each turn to encourage efficiency

## Next Steps

1. **Train Longer**: Increase `total_timesteps` to 1M+ for better strategy
2. **Tune Rewards**: Adjust `REWARD_CONFIG` to encourage specific behaviors
3. **Better Opponents**: Implement self-play with previous checkpoints
4. **Hyperparameter Tuning**: Experiment with learning rate, batch size, etc.
5. **Evaluation Metrics**: Add detailed analysis (action distribution, game length)

## References

- **Detailed Documentation**: See `eng_plans/rl_implementation_summary.md`
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **MaskablePPO**: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- **Gymnasium**: https://gymnasium.farama.org/

## Notes

- Models and logs are gitignored (see `.gitignore`)
- Training is deterministic with fixed seeds
- Environment uses action masking - invalid actions should never occur
- State encoding is fixed-size (206 dims) for neural network compatibility

---

**Last Updated**: 2025-10