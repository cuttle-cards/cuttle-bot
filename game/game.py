from typing import List
from card import Card, Suit, Rank
from game_state import GameState
import uuid
import random

class Game:
    """
    A class that represents a game of Cuttle.

    """
    game_state: GameState
    players: List[int]

    def __init__(self):

        # Initialize the game state
        # randomly shuffle the deck
        deck = self.generate_shuffled_deck()
        self.players = [0,1]

        # deal the cards to players
        hands = self.deal_cards(deck)
        fields = [[], []]

        self.game_state = GameState(hands, fields, deck[11:], [])

    
    def generate_shuffled_deck(self) -> List[Card]:
        # create a list of all cards
        cards = []
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                id = uuid.uuid4()
                cards.append(Card(id, suit, rank))
        
        # shuffle the cards
        random.shuffle(cards)

        return cards
    
    def deal_cards(self, deck: List[Card]) -> List[List[Card]]:
        # deal the cards to players
        # p0 gets 5 cards, p1 gets 6 cards
        # take turns to deal
        hands = [deck[0:5], deck[5:11]]

        return hands


if __name__ == "__main__":
    game = Game()

    print(game.game_state.deck)
    print(game.game_state.hands[0])
    print(game.game_state.hands[1])

    print(game.game_state.get_legal_actions())
    



