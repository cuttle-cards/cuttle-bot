"""
Main module for the Cuttle card game.

This module implements the main game loop and user interaction logic for the Cuttle card game.
It handles both AI and human players, game state management, and game history logging.
"""

from game.game import Game
from game.ai_player import AIPlayer
from game.input_handler import get_interactive_input
import asyncio
import time
import os
import datetime
import io
import logging
from typing import List, Union, Tuple
from game.utils import log_print

HISTORY_DIR = "game_history"

# Create history directory if it doesn't exist
os.makedirs(HISTORY_DIR, exist_ok=True)


def setup_logging() -> Tuple[logging.Logger, io.StringIO]:
    """Set up logging configuration for game history capture.

    This function configures logging to both console and a string buffer
    for later saving to a file.

    Returns:
        Tuple[logging.Logger, io.StringIO]: A tuple containing:
            - logger: Configured logging object
            - log_stream: StringIO buffer containing log output
    """
    log_stream = io.StringIO()

    formatter = logging.Formatter("%(message)s")

    string_handler = logging.StreamHandler(log_stream)
    string_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("cuttle")
    logger.setLevel(logging.INFO)

    logger.addHandler(string_handler)
    logger.addHandler(console_handler)

    logger.handlers = [string_handler, console_handler]

    return logger, log_stream


def save_game_history(log_output: List[str]) -> None:
    """Save the game history to a timestamped file.

    Args:
        log_output (List[str]): List of log messages to save.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_history_{timestamp}.txt"
    filepath = os.path.join(HISTORY_DIR, filename)

    with open(filepath, "w") as f:
        f.write("\n".join(log_output))

    log_print(f"Game history saved to {filepath}")


def get_yes_no_input(prompt: str) -> bool:
    """Get a yes/no input from the user.

    Args:
        prompt (str): The prompt to display to the user.

    Returns:
        bool: True for yes/y, False for no/n.
    """
    while True:
        response = input(prompt + " (y/n): ").lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'")
        time.sleep(0.1)  # Add small delay to prevent log spam


def get_action_index_from_text_input(player_action: str, actions: List[str]) -> Union[int, None]:
    """Get the index of the action from the text input.

    This function supports both numeric indices and exact text matches.

    Args:
        player_action (str): The action to get the index of. Could be a number in string form 
                            or a string detailing the action.
        actions (List[str]): The list of actions to choose from.

    Returns:
        Union[int, None]: The index of the action, or None if the action is not found.
    """
    if player_action.isdigit() and int(player_action) in range(len(actions)):
        return int(player_action)
    
    for i, action in enumerate(actions):
        if player_action.lower() == str(action).lower():
            return i
    return None


def select_saved_game() -> Union[str, None]:
    """Let user select a saved game from the list.

    Returns:
        Union[str, None]: The filename of the selected game, or None if cancelled or no games found.
    """
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


async def initialize_game(use_ai: bool, ai_player: AIPlayer) -> Game:
    """Initialize a new game or load a saved game.

    Args:
        use_ai (bool): Whether to use AI player.
        ai_player (AIPlayer): The AI player instance if use_ai is True.

    Returns:
        Game: The initialized game instance.
    """
    if get_yes_no_input("Would you like to load a saved game?"):
        filename = select_saved_game()
        if filename:
            try:
                game = Game(load_game=filename, ai_player=ai_player)
                log_print("Game loaded successfully!")
                return game
            except Exception as e:
                log_print(f"Error loading game: {e}")
                log_print("Starting new game instead.")
    
    manual_selection = get_yes_no_input("Would you like to manually select initial cards?")
    print(f"use_ai: {use_ai}")
    game = Game(manual_selection=manual_selection, ai_player=ai_player)
    
    if get_yes_no_input("Would you like to save this initial game state?"):
        save_initial_game_state(game)
    
    return game

def save_initial_game_state(game: Game) -> None:
    """Save the initial game state.

    Args:
        game (Game): The game instance to save.
    """
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

async def handle_player_turn(game: Game, use_ai: bool, ai_player: AIPlayer, actions: List[str]) -> Tuple[str, bool]:
    """Handle a player's turn, either AI or human.

    Args:
        game (Game): The current game instance.
        use_ai (bool): Whether AI player is enabled.
        ai_player (AIPlayer): The AI player instance.
        actions (List[str]): List of available actions.

    Returns:
        Tuple[str, bool]: A tuple containing:
            - str: The chosen action index or "end game"
            - bool: Whether the game should end
    """
    is_ai_turn = use_ai and (
        (game.game_state.resolving_one_off and game.game_state.current_action_player == 1)
        or (not game.game_state.resolving_one_off and game.game_state.turn == 1)
    )

    if is_ai_turn:
        return await handle_ai_turn(game, ai_player, actions)
    else:
        return handle_human_turn(game, actions)

async def handle_ai_turn(game: Game, ai_player: AIPlayer, actions: List[str]) -> Tuple[str, bool]:
    """Handle AI player's turn.

    Args:
        game (Game): The current game instance.
        ai_player (AIPlayer): The AI player instance.
        actions (List[str]): List of available actions.

    Returns:
        Tuple[str, bool]: A tuple containing:
            - str: The chosen action index
            - bool: Whether the game should end (always False for AI)
    """
    log_print("AI is thinking...")
    try:
        chosen_action = await ai_player.get_action(game.game_state, actions)
        action_index = actions.index(chosen_action)
        log_print(f"AI chose: {chosen_action}")
        return str(action_index), False
    except Exception as e:
        log_print(f"AI error: {e}. Defaulting to first action.")
        return "0", False

def handle_human_turn(game: Game, actions: List[str]) -> Tuple[str, bool]:
    """Handle human player's turn.

    Args:
        game (Game): The current game instance.
        actions (List[str]): List of available actions.

    Returns:
        Tuple[str, bool]: A tuple containing:
            - str: The chosen action index or "end game"
            - bool: Whether the game should end
    """
    try:
        action_index = get_interactive_input(
            f"Enter your action for player {game.game_state.current_action_player} ('e' to end game):",
            [f"{i}: {str(action)}" for i, action in enumerate(actions)]
        )
        if action_index == -1:
            return "end game", True
        return str(action_index), False
    except KeyboardInterrupt:
        return "e", True

def process_game_action(game: Game, action_index: int, actions: List[str]) -> Tuple[bool, bool, int]:
    """Process a game action and return the game state.

    Args:
        game (Game): The current game instance.
        action_index (int): The index of the chosen action.
        actions (List[str]): List of available actions.

    Returns:
        Tuple[bool, bool, int]: A tuple containing:
            - bool: Whether the game is over
            - bool: Whether the turn is finished
            - int: The winner index (if game is over)
    """
    log_print(f"Player {game.game_state.current_action_player} chose {actions[action_index]}")
    return game.game_state.update_state(actions[action_index])

def update_game_state(game: Game, turn_finished: bool, use_ai: bool) -> None:
    """Update the game state after an action.

    Args:
        game (Game): The current game instance.
        turn_finished (bool): Whether the current turn is finished.
        use_ai (bool): Whether AI player is enabled.
    """
    if turn_finished:
        game.game_state.resolving_one_off = False
    
    if game.game_state.resolving_one_off:
        game.game_state.next_player()
    else:
        # Hide AI's hand when printing game state if playing against AI
        game.game_state.print_state(hide_player_hand=1 if use_ai else None)
        game.game_state.next_turn()

async def game_loop(game: Game, use_ai: bool, ai_player: AIPlayer) -> int:
    """Main game loop. Returns the winner."""
    game_over = False
    winner = None
    
    while not game_over:
        turn_finished = False
        should_stop = False
        invalid_input_count = 0
        MAX_INVALID_INPUTS = 5

        if game.game_state.turn == 0:
            log_print(f"================ Turn {game.game_state.overall_turn} =================")

        while not turn_finished and not game_over:
            time.sleep(0.1)  # Add small delay to prevent log spam
            
            display_game_state(game)
            actions = game.game_state.get_legal_actions()
            for i, action in enumerate(actions):
                log_print(f"{i}: {action}")

            player_action, is_end_game = await handle_player_turn(game, use_ai, ai_player, actions)
            
            if is_end_game:
                game_over = True
                break

            try:
                action_index = int(player_action)
                if action_index is None:
                    raise ValueError
                
                turn_finished, should_stop, winner = process_game_action(game, action_index, actions)
                
                if should_stop or winner is not None:
                    game_over = True
                    break
                    
            except (ValueError, IndexError):
                log_print("Invalid input, please enter a number")
                invalid_input_count += 1
                if invalid_input_count >= MAX_INVALID_INPUTS:
                    log_print(f"Too many invalid inputs ({MAX_INVALID_INPUTS}). Game terminated.")
                    game_over = True
                    break
                continue

            update_game_state(game, turn_finished, use_ai)

    return winner

def display_game_state(game: Game):
    """Display the current game state."""
    if game.game_state.resolving_one_off:
        log_print(f"Actions for player {game.game_state.current_action_player}:")
    else:
        log_print(f"Actions for player {game.game_state.turn}:")

async def main():
    """Main entry point for the game."""
    logger, log_stream = setup_logging()
    use_ai = get_yes_no_input("Would you like to play against AI (as Player 2)?")
    ai_player = AIPlayer() if use_ai else None

    while True:
        game = await initialize_game(use_ai, ai_player)
        
        log_print("\nStarting game...")
        game.game_state.print_state(hide_player_hand=1 if use_ai else None)

        winner = await game_loop(game, use_ai, ai_player)

        log_print(f"Game over! Winner is player {winner}")
        game.game_state.print_state(hide_player_hand=1 if use_ai else None)

        if get_yes_no_input("Would you like to save the game history?"):
            save_game_history(log_stream.getvalue().splitlines())
        
        keep_playing = use_ai and get_yes_no_input("Would you like to play again with AI?") 
        if not keep_playing:
            break


if __name__ == "__main__":
    asyncio.run(main())
