from game.game import Game
from game.action import ActionType
from game.ai_player import AIPlayer
import asyncio
import time


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
    # Ask if user wants to play against AI
    use_ai = get_yes_no_input("Would you like to play against AI (as Player 2)?")

    # Initialize AI player if requested
    ai_player = AIPlayer() if use_ai else None

    # Ask if user wants to load a saved game
    if get_yes_no_input("Would you like to load a saved game?"):
        filename = select_saved_game()
        if filename:
            try:
                game = Game(load_game=filename)
                print("Game loaded successfully!")
            except Exception as e:
                print(f"Error loading game: {e}")
                print("Starting new game instead.")
                game = None
        else:
            print("Starting new game instead.")
            game = None
    else:
        game = None

    if game is None:
        # Ask if user wants to manually select cards
        manual_selection = get_yes_no_input(
            "Would you like to manually select initial cards?"
        )
        game = Game(manual_selection=manual_selection)

        # Ask if user wants to save the initial game state
        if get_yes_no_input("Would you like to save this initial game state?"):
            while True:
                filename = input("Enter filename to save to (without .json): ")
                if filename:
                    try:
                        game.save_game(filename)
                        print("Game saved successfully!")
                        break
                    except Exception as e:
                        print(f"Error saving game: {e}")
                        if not get_yes_no_input("Would you like to try again?"):
                            break
                else:
                    print("Please enter a valid filename.")

    print("\nStarting game...")
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

        while True:
            time.sleep(0.1)  # Add small delay to prevent log spam
            # get legal actions
            if game.game_state.resolving_one_off:
                print(f"Actions for player {game.game_state.current_action_player}:")
            else:
                print(f"Actions for player {game.game_state.turn}:")

            actions = game.game_state.get_legal_actions()
            for i, action in enumerate(actions):
                print(f"{i}: {action}")

            # Check if it's AI's turn (P1) or if AI needs to respond to one-off
            is_ai_turn = use_ai and (
                (
                    game.game_state.resolving_one_off
                    and game.game_state.current_action_player == 1
                )
                or (not game.game_state.resolving_one_off and game.game_state.turn == 1)
            )

            if is_ai_turn:
                print("AI is thinking...")
                try:
                    chosen_action = await ai_player.get_action(game.game_state, actions)
                    action_index = actions.index(chosen_action)
                    print(f"AI chose: {chosen_action}")
                    player_action = str(action_index)
                except Exception as e:
                    print(f"AI error: {e}. Defaulting to first action.")
                    player_action = "0"
            else:
                player_action = input(
                    f"Enter your action for player {game.game_state.current_action_player} ('e' to end game): "
                )

            # Check for end game input
            if player_action.lower() == "e":
                game_over = True
                break

            # invalid player input
            if not player_action.isdigit() or not int(player_action) in range(
                len(actions)
            ):
                print("Invalid input, please enter a number")
                invalid_input_count += 1
                if invalid_input_count >= MAX_INVALID_INPUTS:
                    print(
                        f"Too many invalid inputs ({MAX_INVALID_INPUTS}). Game terminated."
                    )
                    game_over = True
                    break
                continue

            # Reset invalid input counter after a valid input
            invalid_input_count = 0
            player_action = int(player_action)
            print(
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

    print(f"Game over! Winner is player {winner}")
    # Hide AI's hand in final state if playing against AI
    game.game_state.print_state(hide_player_hand=1 if use_ai else None)


if __name__ == "__main__":
    asyncio.run(main())
