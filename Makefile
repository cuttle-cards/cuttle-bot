# Get the current working directory
CURRENT_DIR := $(shell pwd)

# Add command to run tests
test:
	PYTHONPATH=$(CURRENT_DIR) pytest tests -v

run:
	PYTHONPATH=$(CURRENT_DIR) python main.py