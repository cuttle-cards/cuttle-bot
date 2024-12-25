from typing import List, Dict, Optional
from game.card import Card, Suit, Rank
from game.game_state import GameState
from game.serializer import save_game_state, load_game_state
import uuid
import random
import os
import glob


class Game:
    """
    A class that represents a game of Cuttle.

    """

    game_state: GameState
    players: List[int]
    SAVE_DIR = "test_games"

    def __init__(
        self,
        manual_selection: bool = False,
        load_game: Optional[str] = None,
        test_deck: Optional[List[Card]] = None,
    ):
        self.players = [0, 1]

        # Create save directory if it doesn't exist
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        if load_game:
            self.load_game(load_game)
        elif test_deck is not None:
            # Use the provided test deck
            self.initialize_with_test_deck(test_deck)
        elif manual_selection:
            self.initialize_with_manual_selection()
        else:
            self.initialize_with_random_hands()

    def save_game(self, filename: str):
        """
        Save the current game state to a file in the test_games directory.

        Args:
            filename: Name of the save file (without directory)
        """
        if not filename.endswith(".json"):
            filename += ".json"
        save_path = os.path.join(self.SAVE_DIR, filename)
        save_game_state(self.game_state, save_path)
        print(f"Game saved to {save_path}")

    def load_game(self, filename: str):
        """
        Load a game state from a file in the test_games directory.

        Args:
            filename: Name of the save file (without directory)
        """
        if not filename.endswith(".json"):
            filename += ".json"
        load_path = os.path.join(self.SAVE_DIR, filename)
        self.game_state = load_game_state(load_path)
        print(f"Game loaded from {load_path}")

    @classmethod
    def list_saved_games(cls) -> List[str]:
        """
        List all saved games in the test_games directory.

        Returns:
            List of save file names (without directory path)
        """
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        save_files = glob.glob(os.path.join(cls.SAVE_DIR, "*.json"))
        return [os.path.basename(f) for f in save_files]

    def initialize_with_random_hands(self):
        """Initialize the game with randomly dealt hands."""
        # Initialize the game state
        # randomly shuffle the deck
        deck = self.generate_shuffled_deck()

        # deal the cards to players
        hands = self.deal_cards(deck)
        fields = [[], []]

        self.game_state = GameState(hands, fields, deck[11:], [])

    def initialize_with_manual_selection(self):
        """Initialize the game with manually selected hands."""
        all_cards = self.generate_all_cards()
        available_cards = {str(card): card for card in all_cards}
        hands = [[], []]

        # Display card selection interface for each player
        for player in range(2):
            max_cards = 5 if player == 0 else 6
            print(f"\nSelecting cards for Player {player} (max {max_cards} cards):")

            while len(hands[player]) < max_cards:
                self.display_available_cards(available_cards)
                choice = input(
                    f"Enter card number to select (or 'done' to finish Player {player}'s selection): "
                )

                if choice.lower() == "done":
                    break

                try:
                    card_num = int(choice)
                    cards = list(available_cards.values())
                    if 0 <= card_num < len(cards):
                        selected_card = cards[card_num]
                        hands[player].append(selected_card)
                        del available_cards[str(selected_card)]
                        print(f"Selected: {selected_card}")
                    else:
                        print("Invalid card number")
                except ValueError:
                    print("Invalid input. Please enter a number or 'done'")

        # Fill remaining slots with random cards
        self.fill_remaining_slots(hands, available_cards)

        # Create deck from remaining cards
        deck = list(available_cards.values())
        random.shuffle(deck)

        # Initialize game state with empty fields for both players
        fields = [[], []]  # Initialize empty fields for both players
        self.game_state = GameState(hands, fields, deck, [])

    def display_available_cards(self, available_cards: Dict[str, Card]):
        """Display available cards for selection."""
        print("\nAvailable cards:")
        cards = list(available_cards.values())
        for i, card in enumerate(cards):
            print(f"{i}: {card}")

    def fill_remaining_slots(
        self, hands: List[List[Card]], available_cards: Dict[str, Card]
    ):
        """Fill remaining hand slots with random cards."""
        cards = list(available_cards.values())

        # Fill player 0's hand to 5 cards
        while len(hands[0]) < 5:
            card = random.choice(cards)
            hands[0].append(card)
            del available_cards[str(card)]
            cards.remove(card)
            print(f"Randomly added to Player 0's hand: {card}")

        # Fill player 1's hand to 6 cards
        while len(hands[1]) < 6:
            card = random.choice(cards)
            hands[1].append(card)
            del available_cards[str(card)]
            cards.remove(card)
            print(f"Randomly added to Player 1's hand: {card}")

    def generate_all_cards(self) -> List[Card]:
        """Generate all cards without shuffling."""
        cards = []
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                id = uuid.uuid4()
                cards.append(Card(id, suit, rank))
        return cards

    def generate_shuffled_deck(self) -> List[Card]:
        """Generate a shuffled deck of cards."""
        cards = self.generate_all_cards()
        random.shuffle(cards)
        return cards

    def deal_cards(self, deck: List[Card]) -> List[List[Card]]:
        """Deal cards to players."""
        # deal the cards to players
        # p0 gets 5 cards, p1 gets 6 cards
        # take turns to deal
        hands = [deck[0:5], deck[5:11]]
        return hands

    def initialize_with_test_deck(self, test_deck: List[Card]):
        """Initialize the game with a predefined deck for testing."""
        # Deal cards from the test deck
        hands = self.deal_cards(test_deck)
        fields = [[], []]
        self.game_state = GameState(hands, fields, test_deck[11:], [])
