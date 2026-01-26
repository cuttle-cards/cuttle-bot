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

# Dockerized dev environment (backend + Vite)
dev:
	docker compose -f docker-compose.dev.yaml up --build -d

dev-down:
	docker compose -f docker-compose.dev.yaml down

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

debug-rl:
	@echo "Running RL games with detailed logging..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/debug_gameplay.py

analyze-rl:
	@echo "Analyzing RL gameplay logs..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/analyze_logs.py

view-rl:
	@echo "Viewing RL gameplay logs..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/view_game.py

hypersearch-rl:
	@echo "Running hyperparameter search (full)..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/hyperparameter_search.py

hypersearch-quick-rl:
	@echo "Running quick hyperparameter search..."
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/hyperparameter_search.py --quick

compare-rl:
	@echo "Compare experiment results..."
	@echo "Usage: make compare-rl DIR=rl/experiments/20260125_120000"
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR not specified"; \
		exit 1; \
	fi
	source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/compare_experiments.py $(DIR)

monitor-rl:
	@source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/monitor.py

watch-rl:
	@source $(VENV_NAME)/bin/activate && PYTHONPATH=$(CURRENT_DIR) python rl/monitor.py --watch
