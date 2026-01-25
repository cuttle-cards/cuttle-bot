# cuttle-bot


# Set Up
## Local dev (no Docker)
## Create a virtual environment

```bash
python3 -m venv cuttle-bot-3.12
source ./cuttle-bot-3.12/bin/activate
```

Or use the Makefile helper (requires `python3.12` on PATH):

```bash
make setup
```

## Install requirements

```bash
pip install -r requirements.txt
```

## Run the dev servers

Backend API (FastAPI + reload):

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Frontend (Vite):

```bash
cd web && npm run dev
```

Open http://localhost:5173

## Docker dev (hot reload)

```bash
make dev
```

Open http://localhost:5173 (API at http://localhost:8000).

## Set up AI player

The game currently supports two types of AI players: RL based AI, and LLM based AI.

### RL based AI

The repo comes with a model zip file which is loaded into the game server. 

The model can be trained locally (see later sections). 

### LLM Based AI

The AI player uses ollama to generate actions. You'll need to install ollama and set up a model.

Follow the installation guide here: https://github.com/ollama/ollama

The AI player uses the `llama3.2` model. but you can use any other model that supports `chat` mode.



## Run tests

The test output can be quite verbose, so it's recommended to redirect the output to a file.

`tmp.txt` is added to `.gitignore` to avoid polluting the repo with test output.

```bash
source ./cuttle-bot-3.12/bin/activate && make test > tmp.txt 2>&1
```

Or you can simply run `make test` to run the tests and see the output in the terminal.

## run game

```bash
make run
```

## RL Training

Train a Cuttle AI using Reinforcement Learning with **MaskablePPO** and action masking for efficient learning.

### Quick Start

Train an agent (15-30 minutes on modern CPU):
```bash
make train-rl
```

Evaluate the trained agent:
```bash
make eval-rl
```

Monitor training progress:
```bash
make tensorboard
# Open http://localhost:6006 in browser
```

### Quick Test

Run a quick 10K timestep training test (~2-3 minutes):
```bash
make test-rl
```

### How It Works

This implementation uses **action masking** to restrict the model to only legal actions:
- Each turn, the environment provides a mask of valid actions
- The model only considers masked (legal) actions when deciding
- This leads to faster training and better performance than penalty-based approaches
- No wasted training time learning what's illegal

### Configuration

Adjust training parameters in `rl/config.py`:
- Timesteps, learning rate, batch size
- Reward structure (win/loss/progress rewards)
- Environment settings

### Architecture

- `rl/cuttle_env.py`: Gymnasium environment with action masking support (220 lines)
- `rl/self_play_env.py`: Self-play wrapper with masked opponent (110 lines)
- `rl/train.py`: MaskablePPO training script with checkpoints (90 lines)
- `rl/evaluate.py`: Evaluation with action masking (130 lines)
- `rl/config.py`: Hyperparameters and settings (50 lines)

### Output

- Models saved to: `rl/models/`
- Training logs: `rl/logs/` (view with TensorBoard)
- Checkpoints every 10K timesteps
- Checkpoints are gitignored, but the final model named `cuttle_rl_final.zip` is tracked with version control (git)
