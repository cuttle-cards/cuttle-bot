from __future__ import annotations

from typing import List
from card import Card, Purpose, Rank

class GameState:
    """
    A class that represents the state of the game.

    Attributes:
        hands: List[List[Card]] - The hands of the players.
        fields: List[List[Card]] - The fields of the players.
        deck: List[Card] - The deck of the game.
        discard_pile: List[Card] - The discard pile of the game.
        scores: List[int] - The scores of the players.
        targets: List[int] - The score targets of the players.
        turn: int - Whose turn it is - 0 for p0, 1 for p1.

    """
    def __init__(self, hands: List[List[Card]], fields: List[List[Card]], deck: List[Card], discard_pile: List[Card]):
        """
        Initialize the game state.
        """
        self.hands = hands
        self.fields = fields
        self.deck = deck
        self.discard_pile = discard_pile
        self.turn = 0 # 0 for p0, 1 for p1
        self.status = None

    def is_game_over(self) -> bool:
        return self.winner() is not None
    
    def get_player_score(self, player: int) -> int:
        hand = self.hands[player]
        point_cards = [card for card in hand if card.rank.value <= Rank.TEN.value and card.purpose == Purpose.POINTS]

        return sum([card.rank.value for card in point_cards])
    
    def get_player_target(self, player: int) -> int:
        return self.targets[player]
    
    def is_winner(self, player: int) -> bool:
        return self.scores[player] >= self.targets[player]
    
    def winner(self) -> int | None:
        for player in range(len(self.scores)):
            if self.is_winner(player):
                return player
        return None
    
    def is_stalemate(self) -> bool:
        return self.deck == [] and not self.winner()

    def update_state(self, action):
        # Implement logic to update the game state based on the action taken
        pass

    def get_legal_actions(self):
        # Return a list of legal actions based on the current game state
        pass
