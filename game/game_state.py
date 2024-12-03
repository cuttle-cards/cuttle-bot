from __future__ import annotations

from typing import List
from game.card import Card, Purpose, Rank
from game.action import Action, ActionType

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
        self.resolving_two = False
        self.resolving_one_off = False
    
    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.hands)


    def is_game_over(self) -> bool:
        return self.winner() is not None
    
    def get_player_score(self, player: int) -> int:
        hand = self.hands[player]
        field = self.fields[player]
        point_cards = [card for card in field if card.rank.value[1] <= Rank.TEN.value[1] and card.purpose == Purpose.POINTS]


        return sum([card.point_value() for card in point_cards])
    
    def get_player_target(self, player: int) -> int:
        # kings affect targets
        # 1 king on player's field: target is 14
        # 2 kings on player's field: target is 10
        # 3 kings on player's field: target is 5
        # 4 kings on player's field: target is 0
        # no kings, 21

        kings = [card for card in self.fields[player] if card.rank == Rank.KING]
        num_kings = len(kings)

        if num_kings == 0:
            return 21
        elif num_kings == 1:
            return 14
        elif num_kings == 2:
            return 10
        elif num_kings == 3:
            return 5
        else:
            return 0

    
    def is_winner(self, player: int) -> bool:
        return self.get_player_score(player) >= self.get_player_target(player)
    
    def winner(self) -> int | None:
        for player in range(len(self.hands)):
            if self.is_winner(player):
                return player
        return None
    
    def is_stalemate(self) -> bool:
        return self.deck == [] and not self.winner()

    def update_state(self, action: Action):
        # Implement logic to update the game state based on the action taken

        should_stop = False
        if action.action_type == ActionType.DRAW:
            self.draw_card()
        elif action.action_type == ActionType.POINTS:
            won = self.play_points(action.card)
            if won:
                should_stop = True
                return should_stop, self.winner()
        elif action.action_type == ActionType.SCUTTLE:
            self.scuttle(action.card, action.target)
        
        pass

    def draw_card(self):
        # draw a card from the deck
        self.hands[self.turn].append(self.deck.pop())
    
    def play_points(self, card: Card):
        # play a points card
        self.hands[self.turn].remove(card)
        card.purpose = Purpose.POINTS
        card.played_by = self.turn
        self.fields[self.turn].append(card)

        # check if the player has won
        if self.get_player_score(self.turn) >= self.get_player_target(self.turn):
            print(f"Player {self.turn} has won with {self.get_player_score(self.turn)} points!")
            self.status = "win"
            return True
        return False
    
    def scuttle(self, card: Card, target: Card):
        # scuttle a points card
        card.played_by = self.turn
        self.hands[card.played_by].remove(card)
        card.clear_player_info()
        self.discard_pile.append(card)
        self.fields[target.played_by].remove(target)
        target.clear_player_info()
        self.discard_pile.append(target)
    

    def get_legal_actions(self):
        """
        Get the legal actions for the current player.

        Returns:
            List[str] - The legal actions for the current player.
        """
        player = self.turn

        actions = []

        hand = self.hands[player]
        opponent_fields = self.fields[:player] + self.fields[player + 1:]

        if len(hand) < 8 and len(self.deck) > 0:
            draw_action = Action(ActionType.DRAW, None, None)
            actions.append(draw_action)
        
        point_cards = [card for card in hand if card.is_point_card()]
        face_cards = [card for card in hand if card.is_face_card()]

        for card in point_cards:
            actions.append(Action(ActionType.POINTS, card, None))
        
        for card in hand:
            actions.append(f"Play {card} as one-off")

        # scuttle
        for point_card in point_cards:
            for opponent_field , opponent_player in zip(opponent_fields, range(len(opponent_fields))):
                opponent_field_point_cards = [card for card in opponent_field if card.is_point_card()]
                for opponent_point_card in opponent_field_point_cards:
                    if point_card.point_value() > opponent_point_card.point_value() or (point_card.point_value() == opponent_point_card.point_value() and point_card.suit_value()> opponent_point_card.suit_value()):
                        actions.append(Action(ActionType.SCUTTLE, point_card, opponent_point_card))
        

        for card in face_cards:
            actions.append(f"Play {card} as face card")
        
        return actions
    
    def print_state(self):
        print(f"Player {self.turn}'s turn")
        print(f"Deck: {len(self.deck)}")
        print(f"Discard Pile: {len(self.discard_pile)}")
        print("Points: " )
        for i, hand in enumerate(self.hands):
            points = self.get_player_score(i)
            print(f"Player {i}: {points}")
        for i, hand in enumerate(self.hands):
            print(f"Player {i}'s hand: {hand}")
        for i, field in enumerate(self.fields):
            print(f"Player {i}'s field: {field}")
        
