"""Interactive viewer for logged RL games."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def format_card(card_dict: Dict[str, Any]) -> str:
    """Format card dict as readable string."""
    if not card_dict:
        return "None"
    return f"{card_dict['rank']}♦♥♠♣"[0] if card_dict['suit'] == 'DIAMONDS' else \
           f"{card_dict['rank']}♥" if card_dict['suit'] == 'HEARTS' else \
           f"{card_dict['rank']}♠" if card_dict['suit'] == 'SPADES' else \
           f"{card_dict['rank']}♣"


def display_game(game_file: Path) -> None:
    """Display a game log in a readable format."""
    with open(game_file, "r") as f:
        game = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"GAME {game['game_id']}")
    print(f"{'='*80}")
    print(f"Start: {game['start_time']}")
    print(f"Outcome: {game['outcome']['reason'].upper()}")
    if game['outcome']['winner'] is not None:
        print(f"Winner: Player {game['outcome']['winner']}")
    print(f"Total Steps: {game['outcome']['total_steps']}")
    print(f"Final Scores: P0={game['outcome']['final_scores']['player_0']}, "
          f"P1={game['outcome']['final_scores']['player_1']}")
    print(f"{'='*80}\n")
    
    # Display step by step
    for i, step in enumerate(game['steps'][:50]):  # Show first 50 steps
        player = step['player']
        action = step['action']
        state = step['state']
        
        print(f"Step {step['step']:3d} | P{player} | {action['type']:15s}", end="")
        
        if action['card']:
            print(f" | Card: {format_card(action['card'])}", end="")
        if action['target']:
            print(f" | Target: {format_card(action['target'])}", end="")
        
        print(f" | Score: P0={state['scores']['player_0']:2d} P1={state['scores']['player_1']:2d}", end="")
        print(f" | Hands: P0={state['hand_sizes']['player_0']} P1={state['hand_sizes']['player_1']}", end="")
        print(f" | Deck: {state['deck_size']:2d}")
        
        # Show field state every 10 steps
        if (i + 1) % 10 == 0:
            print(f"        {'─'*72}")
            p0_field = [format_card(c) for c in state['field_cards']['player_0']]
            p1_field = [format_card(c) for c in state['field_cards']['player_1']]
            print(f"        P0 Field: {', '.join(p0_field) if p0_field else '(empty)'}")
            print(f"        P1 Field: {', '.join(p1_field) if p1_field else '(empty)'}")
            print(f"        {'─'*72}")
    
    if len(game['steps']) > 50:
        print(f"\n... ({len(game['steps']) - 50} more steps) ...\n")
        
        # Show last 10 steps
        print(f"{'─'*80}")
        print("LAST 10 STEPS:")
        print(f"{'─'*80}\n")
        for step in game['steps'][-10:]:
            player = step['player']
            action = step['action']
            state = step['state']
            
            print(f"Step {step['step']:3d} | P{player} | {action['type']:15s}", end="")
            if action['card']:
                print(f" | {format_card(action['card'])}", end="")
            print(f" | Score: P0={state['scores']['player_0']:2d} P1={state['scores']['player_1']:2d}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main function to view game logs."""
    log_dir = Path("rl/gameplay_logs")
    
    if not log_dir.exists():
        print("❌ No logs found. Run 'make debug-rl' first.")
        return
    
    game_files = sorted(log_dir.glob("game_*.json"))
    
    if not game_files:
        print("❌ No game logs found.")
        return
    
    print(f"\nFound {len(game_files)} games:")
    for i, game_file in enumerate(game_files):
        print(f"  {i}: {game_file.name}")
    
    print("\nEnter game number to view (or 'all' for all games, 'q' to quit):")
    
    while True:
        choice = input("> ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'all':
            for game_file in game_files:
                display_game(game_file)
            break
        else:
            try:
                game_num = int(choice)
                if 0 <= game_num < len(game_files):
                    display_game(game_files[game_num])
                else:
                    print(f"Invalid game number. Choose 0-{len(game_files)-1}")
            except ValueError:
                print("Invalid input. Enter a number, 'all', or 'q'")


if __name__ == "__main__":
    main()
