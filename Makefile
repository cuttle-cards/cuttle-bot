# Get the current working directory
CURRENT_DIR := $(shell pwd)

# Add command to run tests
# --capture=tee-sys is used to capture the output of the tests and print it to the console
test:
	PYTHONPATH=$(CURRENT_DIR) pytest tests -v --capture=tee-sys

run:
	PYTHONPATH=$(CURRENT_DIR) python main.py

# Generate documentation using pdoc
docs:
	PYTHONPATH=$(CURRENT_DIR) python docs.py

# Clean generated documentation
clean-docs:
	rm -rf docs/