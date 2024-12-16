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
        self.last_action_played_by = None
        self.current_action_player = self.turn
        self.status = None
        self.resolving_two = False
        self.resolving_one_off = False
        self.one_off_card_to_counter = None
    
    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.hands)
        self.current_action_player = self.turn
    
    def next_player(self):
        self.current_action_player = (self.current_action_player + 1) % len(self.hands)

    def is_game_over(self) -> bool:
        return self.winner() is not None
    
    def get_player_score(self, player: int) -> int:
        hand = self.hands[player]
        field = self.fields[player]
        point_cards = [card for card in field if card.point_value() <= Rank.TEN.value[1] and card.purpose == Purpose.POINTS]


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
        """
        Returns:
            Tuple[bool, bool, int | None] 
              - Whether the turn is over, 
              - Whether the turn is finished, and 
              - The winner if the game is over.
        """
        # Implement logic to update the game state based on the action taken

        turn_finished = False
        should_stop = False
        winner = None

        if action.action_type == ActionType.DRAW:
            self.draw_card()
            turn_finished = True
            return turn_finished, should_stop, winner
        elif action.action_type == ActionType.POINTS:
            won = self.play_points(action.card)
            turn_finished = True
            if won:
                should_stop = True
                winner = self.winner()
                return turn_finished, should_stop, winner
        elif action.action_type == ActionType.SCUTTLE:
            self.scuttle(action.card, action.target)
        elif action.action_type == ActionType.ONE_OFF:
            turn_finished, played_by = self.play_one_off(self.turn, action.card, None, None)
            if turn_finished:
                should_stop = False
                winner = self.winner()
                return turn_finished, should_stop, winner
            self.resolving_one_off = True
            self.one_off_card_to_counter = action.card
            return turn_finished, should_stop, winner
        elif action.action_type == ActionType.COUNTER:
            turn_finished, played_by = self.play_one_off(self.turn, action.target, action.card, None)
            if turn_finished:
                should_stop = False
                winner = self.winner()
                return turn_finished, should_stop, winner
        elif action.action_type == ActionType.RESOLVE:
            turn_finished, played_by = self.play_one_off(self.turn, action.target, None, action.played_by)
            if turn_finished:
                should_stop = False
                winner = self.winner()
                return turn_finished, should_stop, winner
            
        
        return turn_finished, should_stop, winner

    def draw_card(self, count: int = 1):
        """
        Draw a card from the deck.

        Args:
            count (int): The number of cards to draw. Defaults to 1. If played a 5, draw 2 cards.
        """
        # if player has 8 cards, raise exception
        if len(self.hands[self.turn]) == 8:
            raise Exception("Player has 8 cards, cannot draw")
        # draw a card from the deck
        for _ in range(count):
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

    def play_one_off(self, player: int, card: Card, countered_with: Card = None, last_resolved_by: int = None):
        """
        Play a one-off card.
        """
        # Initial play requires additional input
        self.last_action_played_by = player
        if player == self.turn and countered_with is None and last_resolved_by is None:
            return False, None
        
        # countered with a Two, needs the other player to counter/resolve
        if countered_with is not None:
            if card.point_value() != 2:
                raise Exception("Counter must be a 2")
            if countered_with.purpose != Purpose.COUNTER:
                raise Exception(f"Counter must be with a purpose of counter, instead got {card.purpose}")
            
            played_by = countered_with.played_by
            self.hands[played_by].remove(countered_with)
            self.discard_pile.append(countered_with)
            countered_with.clear_player_info()
            self.last_action_played_by = played_by
            return False, played_by
        else:
            # No counter
            # If last action was opponent (resolve)
            # the one off is played
            # and turn finishes

            self.last_action_played_by = player
            
            if last_resolved_by != player:
                self.hands[self.turn].remove(card)
                card.purpose = Purpose.ONE_OFF
                self.apply_one_off_effect(card)
            else:
            # Last action was player (resolve)
            # countered by opponent
            # the one off is not played
            # but the card is moved to scrap since it was countered
                self.hands[self.turn].remove(card)
                self.discard_pile.append(card)
                card.clear_player_info()
            
            return True, None

        return True, None


    def apply_one_off_effect(self, card: Card):
        if card.rank == Rank.FIVE:
            self.draw_card(2)


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

        if self.resolving_one_off:
            for card in hand:
                if card.rank == Rank.TWO:
                    actions.append(Action(action_type=ActionType.COUNTER, card=card, target=self.one_off_card_to_counter, played_by=self.current_action_player))
            
            
            actions.append(Action(action_type=ActionType.RESOLVE, card=None, target=self.one_off_card_to_counter, played_by=self.current_action_player))
            return actions

        if len(hand) < 8 and len(self.deck) > 0:
            draw_action = Action(action_type=ActionType.DRAW, card=None, target=None, played_by=self.current_action_player)
            actions.append(draw_action)
        
        point_cards = [card for card in hand if card.is_point_card()]
        face_cards = [card for card in hand if card.is_face_card()]

        for card in point_cards:
            actions.append(Action(action_type=ActionType.POINTS, card=card, target=None, played_by=self.current_action_player))
        
        for card in hand:
            # Untargeted one-off
            if card.rank in [Rank.ACE, Rank.FIVE, Rank.SIX]:
                actions.append(Action(action_type=ActionType.ONE_OFF, card=card, target=None, played_by=self.current_action_player))
        
        # scuttle
        for point_card in point_cards:
            for opponent_field , opponent_player in zip(opponent_fields, range(len(opponent_fields))):
                opponent_field_point_cards = [card for card in opponent_field if card.is_point_card()]
                for opponent_point_card in opponent_field_point_cards:
                    if point_card.point_value() > opponent_point_card.point_value() or (point_card.point_value() == opponent_point_card.point_value() and point_card.suit_value()> opponent_point_card.suit_value()):
                        actions.append(Action(action_type=ActionType.SCUTTLE, card=point_card, target=opponent_point_card, played_by=self.current_action_player))
        

        for card in face_cards:
            actions.append(f"Play {card} as face card")
        
        return actions
    
    def print_state(self):
        print("--------------------------------")
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
        print("--------------------------------")
        
