from game.game import Game

if __name__ == "__main__":
    game = Game()

    print(game.game_state.deck)
    print(game.game_state.hands[0])
    print(game.game_state.hands[1])

    print(game.game_state.get_legal_actions())

    

    # wait for user input
    # will be the index of the action in the list of actions
    game_over = False
    while not game_over:
        # initialize variables
        # Assume the turn would finished with one move
        turn_finished = False
        should_stop = False
        winner = None

        while True:
            # get legal actions
            if game.game_state.resolving_one_off:
                print(f"Actions for player {game.game_state.current_action_player}:")
            else:
                print(f"Actions for player {game.game_state.turn}:")

            actions = game.game_state.get_legal_actions()
            for i, action in enumerate(actions):
                print(f"{i}: {action}")

            player_action = input(f"Enter your action for player {game.game_state.current_action_player}: ")
            
            print(player_action, type(player_action), player_action in range(len(actions)))
            # invalid player input
            if not player_action.isdigit() or not int(player_action) in range(len(actions)):
                print("Invalid input, please enter a number")
                continue
            

            player_action = int(player_action)
            print(f"You chose {actions[player_action]}")
            turn_finished, should_stop, winner = game.game_state.update_state(actions[player_action])
            if should_stop:
                game_over, winner = True, winner
                break
                
            if turn_finished:
                game.game_state.resolving_one_off = False
                break

            if game.game_state.resolving_one_off:
                game.game_state.next_player()
        
        
        game.game_state.print_state()
        game.game_state.next_turn()
    
    print(f"Game over! Winner is player {winner}")
    game.game_state.print_state()
    
