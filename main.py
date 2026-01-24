"""
Main module for the Cuttle card game.

This module implements the main game loop and user interaction logic for the Cuttle card game.
It handles both AI and human players, game state management, and game history logging.
"""

import asyncio
import datetime
import io
import logging
import os
import time
from typing import List, Optional, Tuple, Union

from game.action import Action  # Import Action
from game.ai_player import AIPlayer
from game.game import Game
from game.input_handler import get_interactive_input
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
        print(f"{prompt} response: {response}")
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        print(f"{prompt} Please enter 'y' or 'n'")
        time.sleep(0.05)  # Add small delay to prevent log spam


def get_action_from_text_input(
    player_action: str, actions: List[Action]
) -> Optional[Action]:
    """Get the Action object from the text input.

    This function supports both numeric indices and exact text matches.

    Args:
        player_action (str): The action to get the index of. Could be a number in string form
                            or a string detailing the action.
        actions (List[Action]): The list of Action objects to choose from.

    Returns:
        Optional[Action]: The chosen Action object, or None if the action is not found.
    """
    if player_action.isdigit():
        try:
            index = int(player_action)
            if 0 <= index < len(actions):
                return actions[index]
        except ValueError:
            pass  # Fall through to check text match

    action_str = player_action.lower()
    for action in actions:
        if action_str == str(action).lower():
            return action
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


async def initialize_game(use_ai: bool, ai_player: Optional[AIPlayer]) -> Game:
    """Initialize a new game or load a saved game.

    Args:
        use_ai (bool): Whether to use AI player.
        ai_player (Optional[AIPlayer]): The AI player instance if use_ai is True.

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

    manual_selection = get_yes_no_input(
        "Would you like to manually select initial cards?"
    )
    log_print(f"use_ai: {use_ai}")
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


async def handle_player_turn(
    game: Game, use_ai: bool, ai_player: Optional[AIPlayer], actions: List[Action]
) -> Tuple[Optional[Action], bool]:
    """Handle a player's turn, either AI or human.

    Args:
        game (Game): The current game instance.
        use_ai (bool): Whether AI player is enabled.
        ai_player (Optional[AIPlayer]): The AI player instance.
        actions (List[Action]): List of available Action objects.

    Returns:
        Tuple[Optional[Action], bool]: A tuple containing:
            - Optional[Action]: The chosen Action object or None to end game
            - bool: Whether the game should end
    """
    is_ai_turn = (
        use_ai
        and ai_player is not None
        and (
            (
                game.game_state.resolving_one_off
                and game.game_state.current_action_player == 1
            )
            or (not game.game_state.resolving_one_off and game.game_state.turn == 1)
        )
    )

    if is_ai_turn:
        # Check ai_player is not None before calling handle_ai_turn
        # Assert that ai_player is not None, satisfying mypy
        assert ai_player is not None, (
            "AI turn triggered but ai_player is None. This should not happen."
        )
        return await handle_ai_turn(game, ai_player, actions)
    else:
        return handle_human_turn(game, actions)


async def handle_ai_turn(
    game: Game, ai_player: AIPlayer, actions: List[Action]
) -> Tuple[Optional[Action], bool]:
    """Handle AI player's turn.

    Args:
        game (Game): The current game instance.
        ai_player (AIPlayer): The AI player instance.
        actions (List[Action]): List of available Action objects.

    Returns:
        Tuple[Optional[Action], bool]: A tuple containing:
            - Optional[Action]: The chosen Action object
            - bool: Whether the game should end (always False for AI)
    """
    log_print("AI is thinking...")
    try:
        chosen_action = await ai_player.get_action(game.game_state, actions)
        log_print(f"AI chose: {chosen_action}")
        return chosen_action, False
    except Exception as e:
        log_print(f"AI error: {e}. Defaulting to first action.")
        return actions[0] if actions else None, False


def handle_human_turn(
    game: Game, actions: List[Action]
) -> Tuple[Optional[Action], bool]:
    """Handle human player's turn.

    Args:
        game (Game): The current game instance.
        actions (List[Action]): List of available Action objects.

    Returns:
        Tuple[Optional[Action], bool]: A tuple containing:
            - Optional[Action]: The chosen Action object or None to end game
            - bool: Whether the game should end
    """
    action_strs = [
        str(action) for action in actions
    ]  # Convert actions to strings for display

    # Use try-except for KeyboardInterrupt
    try:
        chosen_action_index = get_interactive_input(
            f"Enter your action for player {game.game_state.current_action_player} ('e' to end game):",
            action_strs,
        )

        if chosen_action_index == -1:  # Indicates 'end game' or cancellation
            return None, True

        # Check if the returned index is valid for the current actions list
        if 0 <= chosen_action_index < len(actions):
            chosen_action = actions[chosen_action_index]
            return chosen_action, False
        else:
            log_print(
                f"Invalid action index received: {chosen_action_index}. Please try again."
            )
            # Indicate failure to choose a valid action this turn
            # Let the loop retry
            return None, False  # Returning None action, but game does not end

    except KeyboardInterrupt:
        # Handle Ctrl+C
        log_print("\nGame interrupted by user (Ctrl+C). Ending game.")
        return None, True


def process_game_action(game: Game, action: Action) -> Tuple[bool, bool, Optional[int]]:
    """Process the chosen game action.

    Args:
        game (Game): The current game instance.
        action (Action): The Action object chosen by the player.

    Returns:
        Tuple[bool, bool, Optional[int]]: A tuple containing:
            - bool: Whether the turn finished
            - bool: Whether the game ended
            - Optional[int]: The winner's index, if any
    """
    # Pass the Action object directly to update_state
    turn_finished, turn_ended, winner = game.game_state.update_state(action)
    return turn_finished, turn_ended, winner


def update_game_state(game: Game, turn_finished: bool, use_ai: bool) -> None:
    """Update game state after an action (draw card, switch turn).

    Args:
        game (Game): The current game instance.
        turn_finished (bool): Whether the current player's turn is finished.
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


async def game_loop(
    game: Game, use_ai: bool, ai_player: Optional[AIPlayer]
) -> Optional[int]:
    """Main game loop. Returns the winner."""
    game_over = False
    winner = None

    while not game_over:
        turn_finished = False
        should_stop = False
        invalid_input_count = 0
        MAX_INVALID_INPUTS = 5

        if game.game_state.turn == 0:
            log_print(
                f"================ Turn {game.game_state.overall_turn} ================="
            )

        # Moved print_state out of the inner loop
        # Display state once per player's attempt cycle
        display_game_state(game)  # Includes printing available actions

        while not turn_finished and not game_over:
            # Get legal actions for the current state
            actions: List[Action] = game.game_state.get_legal_actions()

            # Check for no actions (should be rare)
            if not actions:
                log_print(
                    f"Player {game.game_state.current_action_player} has no legal actions!"
                )
                if not game.game_state.deck:
                    log_print("Deck empty and no actions. Ending turn.")
                    game.game_state.next_turn()
                    # Break inner loop to re-evaluate outer loop condition (stalemate/game over)
                    break
                else:
                    log_print(
                        "Error: No legal actions but deck is not empty. Skipping turn."
                    )
                    game.game_state.next_turn()
                    # Break inner loop to re-evaluate state
                    break

            # Print actions only if human turn
            if not (use_ai and game.game_state.current_action_player == 1):
                log_print("Available actions:")
                for i, action in enumerate(actions):
                    log_print(f"{i}: {action}")

            # Handle turn
            chosen_action, is_end_game = await handle_player_turn(
                game, use_ai, ai_player, actions
            )

            if is_end_game:
                log_print("Game ended by player.")
                game_over = True
                break  # Break inner loop

            if chosen_action is None:  # Human entered invalid input or AI failed
                log_print("Invalid input received. Please try again.")
                invalid_input_count += 1
                if invalid_input_count >= MAX_INVALID_INPUTS:
                    log_print(
                        f"Too many invalid inputs ({MAX_INVALID_INPUTS}). Game terminated."
                    )
                    game_over = True
                    break  # Break inner loop
                continue  # Retry input in the inner loop

            # Reset invalid count on valid action
            invalid_input_count = 0

            # Process the valid action
            try:
                turn_finished, should_stop, winner_result = process_game_action(
                    game, chosen_action
                )

                if should_stop:
                    game_over = True
                    winner = winner_result
                    break  # Break inner loop

                # Update game state (draw, switch turn) only if the turn finished
                update_game_state(game, turn_finished, use_ai)

            except Exception as e:
                # Catch potential errors during action processing
                log_print(f"Error processing action '{chosen_action}': {e}")
                log_print("Attempting to recover or end turn.")
                # Decide recovery strategy: maybe force draw or end turn?
                # For now, just end the turn to avoid infinite loops
                turn_finished = True
                update_game_state(game, turn_finished, use_ai)
                # Continue to next turn in the outer loop
                break  # Break inner loop

        # After inner loop: check if game ended or continue outer loop
        if game.game_state.is_game_over():
            winner = game.game_state.winner()
            game_over = True
        elif game.game_state.is_stalemate():
            log_print("Stalemate detected!")
            game_over = True
            winner = None  # Indicate stalemate

    # Final state display after loop ends
    display_game_state(game)
    return winner


def display_game_state(game: Game) -> None:
    """Display the current game state."""
    hide_hand = 1 if game.game_state.use_ai else None
    game.game_state.print_state(hide_player_hand=hide_hand)


async def main() -> None:
    """Main entry point for the game."""
    logger, log_stream = setup_logging()
    use_ai = get_yes_no_input(
        "Would you like to play against AI (as Player 1)?"
    )  # Changed to Player 1
    ai_player = AIPlayer() if use_ai else None

    while True:
        # Pass Optional[AIPlayer] to initialize_game
        game = await initialize_game(use_ai, ai_player)

        log_print("\nStarting game...")
        # display_game_state(game) # Initial display happens in game_loop

        # Pass Optional[AIPlayer] to game_loop
        winner = await game_loop(game, use_ai, ai_player)

        if winner is not None:
            log_print(f"Game over! Winner is player {winner}")
        else:
            log_print("Game over! Ended by player or Stalemate.")

        # game.game_state.print_state(hide_player_hand=1 if use_ai else None)

        if get_yes_no_input("Would you like to save the game history?"):
            save_game_history(log_stream.getvalue().splitlines())

        # Changed condition to check if AI was used for replay prompt
        keep_playing = use_ai and get_yes_no_input(
            "Would you like to play again with AI?"
        )
        if not keep_playing:
            break


if __name__ == "__main__":
    asyncio.run(main())
