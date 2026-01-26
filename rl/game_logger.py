"""Logger for detailed RL gameplay analysis."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from game.action import Action
from game.card import Card
from game.game import Game


class GameplayLogger:
    """Logs detailed gameplay information for debugging RL agents."""
    
    def __init__(self, log_dir: str = "rl/gameplay_logs"):
        """Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_game: Optional[Dict[str, Any]] = None
        self.games_logged = 0
        self.max_games_per_session = 10  # Only log first 10 games per training
        
    def start_game(self, game: Game) -> None:
        """Start logging a new game."""
        if self.games_logged >= self.max_games_per_session:
            return  # Don't log more than max games
            
        self.current_game = {
            "game_id": self.games_logged,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "outcome": None,
            "step_count": 0,
        }
        
    def log_step(
        self,
        step_num: int,
        player: int,
        action: Action,
        game: Game,
        reward: float,
        legal_action_count: int,
    ) -> None:
        """Log a single step of gameplay."""
        if self.current_game is None:
            return
            
        step_info = {
            "step": step_num,
            "player": player,
            "action": {
                "type": action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                "card": self._card_to_dict(action.card) if action.card else None,
                "target": self._card_to_dict(action.target) if action.target else None,
            },
            "reward": float(reward),
            "legal_actions_count": legal_action_count,
            "state": self._get_game_state_snapshot(game, player),
        }
        
        self.current_game["steps"].append(step_info)
        self.current_game["step_count"] = step_num
        
    def end_game(
        self,
        game: Game,
        winner: Optional[int],
        reason: str,
        step_count: int,
    ) -> None:
        """End current game and save log."""
        if self.current_game is None:
            return
            
        self.current_game["outcome"] = {
            "winner": winner,
            "reason": reason,
            "total_steps": step_count,
            "final_scores": {
                "player_0": game.game_state.get_player_score(0),
                "player_1": game.game_state.get_player_score(1),
            },
            "final_targets": {
                "player_0": game.game_state.get_player_target(0),
                "player_1": game.game_state.get_player_target(1),
            },
        }
        
        # Save to file
        filename = f"game_{self.games_logged:03d}_{reason}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.current_game, f, indent=2)
            
        print(f"📝 Saved gameplay log: {filepath}")
        self.games_logged += 1
        self.current_game = None
        
    def _card_to_dict(self, card: Card) -> Dict[str, Any]:
        """Convert card to dictionary."""
        return {
            "rank": card.rank.name,
            "suit": card.suit.name,
            "display": str(card),
        }
        
    def _get_game_state_snapshot(self, game: Game, current_player: int) -> Dict[str, Any]:
        """Get snapshot of current game state."""
        return {
            "current_player": current_player,
            "scores": {
                "player_0": game.game_state.get_player_score(0),
                "player_1": game.game_state.get_player_score(1),
            },
            "hand_sizes": {
                "player_0": len(game.game_state.hands[0]),
                "player_1": len(game.game_state.hands[1]),
            },
            "field_cards": {
                "player_0": [self._card_to_dict(c) for c in game.game_state.get_player_field(0)],
                "player_1": [self._card_to_dict(c) for c in game.game_state.get_player_field(1)],
            },
            "deck_size": len(game.game_state.deck),
            "discard_size": len(game.game_state.discard_pile),
            "resolving_one_off": game.game_state.resolving_one_off,
            "resolving_three": game.game_state.resolving_three,
        }
    
    def generate_summary(self) -> None:
        """Generate a summary of all logged games."""
        if self.games_logged == 0:
            print("No games logged yet.")
            return
            
        summary = {
            "total_games": self.games_logged,
            "outcomes": {},
            "avg_steps": 0,
            "timeout_rate": 0,
        }
        
        total_steps = 0
        timeouts = 0
        
        for i in range(self.games_logged):
            for reason in ["timeout", "win", "stalemate"]:
                filepath = self.log_dir / f"game_{i:03d}_{reason}.json"
                if filepath.exists():
                    with open(filepath, "r") as f:
                        game_data = json.load(f)
                        reason = game_data["outcome"]["reason"]
                        summary["outcomes"][reason] = summary["outcomes"].get(reason, 0) + 1
                        total_steps += game_data["outcome"]["total_steps"]
                        if reason == "timeout":
                            timeouts += 1
                    break
        
        if self.games_logged > 0:
            summary["avg_steps"] = total_steps / self.games_logged
            summary["timeout_rate"] = timeouts / self.games_logged
        
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("GAMEPLAY SUMMARY")
        print(f"{'='*60}")
        print(f"Total games logged: {summary['total_games']}")
        print(f"Average steps per game: {summary['avg_steps']:.1f}")
        print(f"Timeout rate: {summary['timeout_rate']*100:.1f}%")
        print("\nOutcomes:")
        for outcome, count in summary["outcomes"].items():
            print(f"  {outcome}: {count} ({count/self.games_logged*100:.1f}%)")
        print(f"{'='*60}\n")
        print(f"📁 Logs saved to: {self.log_dir.absolute()}")
        print(f"{'='*60}\n")
