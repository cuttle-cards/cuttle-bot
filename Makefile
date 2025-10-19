# Get the current working directory
CURRENT_DIR := $(shell pwd)

# Virtual environment name
VENV_NAME := cuttle-bot-3.12

# Add command to run tests
# --capture=tee-sys is used to capture the output of the tests and print it to the console
test:
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) pytest tests -v --capture=tee-sys

run:
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python main.py

run-with-rl:
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python main_with_rl_ai.py
# Generate documentation using pdoc
docs:
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python docs.py

# Clean generated documentation
clean-docs:
	rm -rf docs/

# Setup virtual environment
setup:
	python3.12 -m venv $(VENV_NAME)
	source $(VENV_NAME)/bin/activate && pip install -r requirements.txt

# Clean virtual environment
clean-venv:
	rm -rf $(VENV_NAME)/

# Default target
all: test

# Type checking
typecheck:
	@echo "Running mypy type checks..."
	source $(VENV_NAME)/bin/activate && mypy .

# RL Training commands (with action masking)
train-rl:
	@echo "Training RL agent with MaskablePPO..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/train.py

eval-rl:
	@echo "Evaluating RL agent..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/evaluate.py

tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006"
	@echo "Press Ctrl+C to stop"
	source $(VENV_NAME)/bin/activate && tensorboard --logdir=rl/logs --port=6006

test-rl:
	@echo "Quick RL training test with action masking (10K timesteps, ~2-3 minutes)..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python -c \
		"from rl import config; config.TRAINING_CONFIG['total_timesteps'] = 10000; \
		exec(open('rl/train.py').read())"