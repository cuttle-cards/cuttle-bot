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