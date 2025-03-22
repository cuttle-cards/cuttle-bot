from game.game import Game
from game.ai_player import AIPlayer
from game.input_handler import get_interactive_input
import asyncio
import time
import os
import datetime
import io
import logging
from typing import List, Union
from game.utils import log_print

HISTORY_DIR = "game_history"

# Create history directory if it doesn't exist
os.makedirs(HISTORY_DIR, exist_ok=True)


def setup_logging():
    """Set up logging to capture game history"""
    # Create string IO for capturing output
    log_stream = io.StringIO()

    # Create formatter
    formatter = logging.Formatter("%(message)s")

    # Create string handler
    string_handler = logging.StreamHandler(log_stream)
    string_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Get logger
    logger = logging.getLogger("cuttle")
    logger.setLevel(logging.INFO)

    # Add handlers
    logger.addHandler(string_handler)
    logger.addHandler(console_handler)

    # Clear any existing handlers
    logger.handlers = [string_handler, console_handler]

    return logger, log_stream


def save_game_history(log_output: List[str]):
    """Save game history to a file"""
    # Create game_history directory if it doesn't exist
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_history_{timestamp}.txt"
    filepath = os.path.join(HISTORY_DIR, filename)

    # Write history to file
    with open(filepath, "w") as f:
        f.write("\n".join(log_output))

    log_print(f"Game history saved to {filepath}")


def get_yes_no_input(prompt: str) -> bool:
    """Get a yes/no input from the user."""
    while True:
        response = input(prompt + " (y/n): ").lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'")
        time.sleep(0.1)  # Add small delay to prevent log spam


def get_action_index_from_text_input(player_action: str, actions: List[str]) -> Union[int, None]:
    """Get the index of the action from the text input."""
    # Input as action index
    if player_action.isdigit() and int(player_action) in range(len(actions)):
        return int(player_action)
    
    # Input as action description
    for i, action in enumerate(actions):
        if player_action.lower() == str(action).lower():
            return i
    return None


def select_saved_game() -> str:
    """Let user select a saved game from the list."""
    saved_games = Game.list_saved_games()
    if not saved_games:
        print("No saved games found.")
        return None

    print("\nAvailable saved games:")
    for i, filename in enumerate(saved_games):
        print(f"{i}: {filename}")

    while True:
        try:
            choice = input("Enter the number of the game to load (or 'cancel'): ")
            if choice.lower() == "cancel":
                return None
            index = int(choice)
            if 0 <= index < len(saved_games):
                return saved_games[index]
            print("Invalid number, please try again.")
        except ValueError:
            print("Please enter a number or 'cancel'.")


async def main():
    # Set up logging
    logger, log_stream = setup_logging()

    # Ask if user wants to play against AI
    use_ai = get_yes_no_input("Would you like to play against AI (as Player 2)?")

    # Initialize AI player if requested
    ai_player = AIPlayer() if use_ai else None

    # Ask if user wants to load a saved game
    if get_yes_no_input("Would you like to load a saved game?"):
        filename = select_saved_game()
        if filename:
            try:
                game = Game(load_game=filename, use_ai=use_ai)
                log_print("Game loaded successfully!")
            except Exception as e:
                log_print(f"Error loading game: {e}")
                log_print("Starting new game instead.")
                game = None
        else:
            log_print("Starting new game instead.")
            game = None
    else:
        game = None

    if game is None:
        # Ask if user wants to manually select cards
        manual_selection = get_yes_no_input(
            "Would you like to manually select initial cards?"
        )
        print(f"use_ai: {use_ai}")
        game = Game(manual_selection=manual_selection, use_ai=use_ai)

        # Ask if user wants to save the initial game state
        if get_yes_no_input("Would you like to save this initial game state?"):
            while True:
                filename = input("Enter filename to save to (without .json): ")
                if filename:
                    try:
                        game.save_game(filename)
                        log_print("Game saved successfully!")
                        break
                    except Exception as e:
                        log_print(f"Error saving game: {e}")
                        if not get_yes_no_input("Would you like to try again?"):
                            break
                else:
                    log_print("Please enter a valid filename.")

    log_print("\nStarting game...")
    # Hide AI's hand (player 1) if playing against AI
    game.game_state.print_state(hide_player_hand=1 if use_ai else None)

    game_over = False
    while not game_over:
        # initialize variables
        # Assume the turn would finished with one move
        turn_finished = False
        should_stop = False
        winner = None
        invalid_input_count = 0  # Counter for invalid inputs
        MAX_INVALID_INPUTS = 5  # Maximum number of invalid inputs allowed

        if game.game_state.turn == 0:
            log_print(
                f"================ Turn {game.game_state.overall_turn} ================="
            )

        while True:
            time.sleep(0.1)  # Add small delay to prevent log spam
            # get legal actions
            log_print(f"player: {game.game_state.turn}")
            if game.game_state.resolving_one_off:
                log_print(
                    f"Actions for player {game.game_state.current_action_player}:"
                )
            else:
                log_print(f"Actions for player {game.game_state.turn}:")

            actions = game.game_state.get_legal_actions()
            for i, action in enumerate(actions):
                log_print(f"{i}: {action}")

            # Check if it's AI's turn (P1) or if AI needs to respond to one-off
            is_ai_turn = use_ai and (
                (
                    game.game_state.resolving_one_off
                    and game.game_state.current_action_player == 1
                )
                or (not game.game_state.resolving_one_off and game.game_state.turn == 1)
            )

            if is_ai_turn:
                log_print("AI is thinking...")
                try:
                    chosen_action = await ai_player.get_action(game.game_state, actions)
                    action_index = actions.index(chosen_action)
                    log_print(f"AI chose: {chosen_action}")
                    player_action = str(action_index)
                except Exception as e:
                    log_print(f"AI error: {e}. Defaulting to first action.")
                    player_action = "0"
            else:
                try:
                    action_index = get_interactive_input(
                        f"Enter your action for player {game.game_state.current_action_player} ('e' to end game):",
                        [str(action) for action in actions]
                    )
                    player_action = str(action_index)
                except KeyboardInterrupt:
                    player_action = 'e'

            # Check for end game input
            if player_action.lower() == "e":
                game_over = True
                break

            # Since we're now getting the action index directly, we don't need to parse it
            action_index = int(player_action)

            # invalid player input
            if action_index is None:
                log_print("Invalid input, please enter a number")
                invalid_input_count += 1
                if invalid_input_count >= MAX_INVALID_INPUTS:
                    log_print(
                        f"Too many invalid inputs ({MAX_INVALID_INPUTS}). Game terminated."
                    )
                    game_over = True
                    break
                continue

            # Reset invalid input counter after a valid input
            invalid_input_count = 0
            player_action = action_index
            log_print(
                f"Player {game.game_state.current_action_player} chose {actions[player_action]}"
            )
            turn_finished, should_stop, winner = game.game_state.update_state(
                actions[player_action]
            )

            # Check for game over conditions
            if should_stop or winner is not None:
                game_over = True
                break

            if turn_finished:
                game.game_state.resolving_one_off = False
                break

            if game.game_state.resolving_one_off:
                game.game_state.next_player()

        if game_over:
            break

        # Hide AI's hand when printing game state if playing against AI
        game.game_state.print_state(hide_player_hand=1 if use_ai else None)
        game.game_state.next_turn()

    log_print(f"Game over! Winner is player {winner}")
    # Hide AI's hand in final state if playing against AI
    game.game_state.print_state(hide_player_hand=1 if use_ai else None)

    # Ask if user wants to save game history
    if get_yes_no_input("Would you like to save the game history?"):
        save_game_history(log_stream.getvalue().splitlines())


if __name__ == "__main__":
    asyncio.run(main())
