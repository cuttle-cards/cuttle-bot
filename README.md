# cuttle-bot


# Set Up
## Create a virtual environment

```bash
python3 -m venv cuttle-bot-3.12
source ./cuttle-bot-3.12/bin/activate
```


## Install requirements

```bash
pip install -r requirements.txt
```

## Set up AI player

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
