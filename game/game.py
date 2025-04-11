"""
Game module for the Cuttle card game.

This module provides the main Game class that manages the game state, card dealing,
saving/loading games, and initialization of game sessions. It supports both manual
and automatic card selection, as well as AI players.
"""

from typing import List, Dict, Optional
from game.card import Card, Suit, Rank
from game.game_state import GameState
from game.serializer import save_game_state, load_game_state
import uuid
import random
import os
import glob
import time
import sys
from game.ai_player import AIPlayer


class Game:
    """A class that represents a game of Cuttle.

    This class manages the game state, handles card dealing, game initialization,
    and provides methods for saving and loading game states. It supports both
    manual and automatic card selection, as well as AI players.

    Attributes:
        game_state (GameState): The current state of the game.
        players (List[int]): List of player indices [0, 1].
        SAVE_DIR (str): Directory where game states are saved.
        ai_player (AIPlayer): Optional AI player instance.
        logger: Function to use for logging (defaults to print).
    """

    game_state: GameState
    players: List[int]
    SAVE_DIR = "test_games"
    ai_player: AIPlayer

    def __init__(
        self,
        manual_selection: bool = False,
        load_game: Optional[str] = None,
        test_deck: Optional[List[Card]] = None,
        logger=print,  # Default to print if no logger provided
        ai_player: Optional[AIPlayer] = None,
    ):
        """Initialize a new game of Cuttle.

        Args:
            manual_selection (bool, optional): Whether to allow manual card selection.
                Defaults to False.
            load_game (Optional[str], optional): Filename of a saved game to load.
                Defaults to None.
            test_deck (Optional[List[Card]], optional): Predefined deck for testing.
                Defaults to None.
            logger (callable, optional): Function to use for logging. Defaults to print.
            ai_player (Optional[AIPlayer], optional): AI player instance. Defaults to None.
        """
        self.players = [0, 1]
        self.logger = logger

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
        self.game_state.use_ai = ai_player is not None
        self.ai_player = ai_player
        self.game_state.ai_player = self.ai_player

    def save_game(self, filename: str) -> None:
        """Save the current game state to a file.

        Args:
            filename (str): Name of the save file (without directory).
                If .json extension is not provided, it will be added.

        Note:
            The game state is saved in JSON format in the SAVE_DIR directory.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        save_path = os.path.join(self.SAVE_DIR, filename)
        save_game_state(self.game_state, save_path)
        self.logger(f"Game saved to {save_path}")

    def load_game(self, filename: str) -> None:
        """Load a game state from a file.

        Args:
            filename (str): Name of the save file (without directory).
                If .json extension is not provided, it will be added.

        Note:
            The game state is loaded from a JSON file in the SAVE_DIR directory.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        load_path = os.path.join(self.SAVE_DIR, filename)
        self.game_state = load_game_state(load_path)
        self.logger(f"Game loaded from {load_path}")

    @classmethod
    def list_saved_games(cls) -> List[str]:
        """List all saved games in the test_games directory.

        Returns:
            List[str]: List of save file names (without directory path).

        Note:
            Creates the save directory if it doesn't exist.
        """
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        save_files = glob.glob(os.path.join(cls.SAVE_DIR, "*.json"))
        return [os.path.basename(f) for f in save_files]

    def initialize_with_random_hands(self) -> None:
        """Initialize the game with randomly dealt hands.

        This method:
        1. Generates and shuffles a complete deck of cards
        2. Deals 5 cards to player 0 and 6 cards to player 1
        3. Creates a new GameState with empty fields and the remaining deck
        """
        deck = self.generate_shuffled_deck()
        hands = self.deal_cards(deck)
        fields = [[], []]
        self.game_state = GameState(hands, fields, deck[11:], [])

    def initialize_with_manual_selection(self) -> None:
        """Initialize the game with manual card selection.

        This method allows players to manually select their starting hands:
        - Player 0 can select up to 5 cards
        - Player 1 can select up to 6 cards
        
        Any unselected card slots will be filled randomly.
        The remaining cards form the deck.

        Note:
            In test environment (pytest), card display is suppressed.
        """
        all_cards = self.generate_all_cards()
        available_cards = {str(card): card for card in all_cards}
        hands = [[], []]

        # Manual selection for both players
        for player in range(2):
            max_cards = 5 if player == 0 else 6
            self.logger(
                f"\nSelecting cards for Player {player} (max {max_cards} cards):"
            )

            while len(hands[player]) < max_cards:
                time.sleep(0.1)  # Add small delay to prevent log spam
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
                        self.logger(f"Selected: {selected_card}")
                    else:
                        self.logger("Invalid card number")
                except ValueError:
                    self.logger("Invalid input. Please enter a number or 'done'")

        # Fill remaining slots with random cards
        self.fill_remaining_slots(hands, available_cards)

        # Create deck from remaining cards
        deck = list(available_cards.values())
        random.shuffle(deck)

        # Initialize game state with empty fields for both players
        fields = [[], []]  # Initialize empty fields for both players
        self.game_state = GameState(hands, fields, deck, [], logger=self.logger)

    def display_available_cards(self, available_cards: Dict[str, Card]) -> None:
        """Display available cards for selection.

        Args:
            available_cards (Dict[str, Card]): Dictionary of available cards,
                where keys are string representations of cards and values are Card objects.

        Note:
            Display is suppressed in test environment (pytest).
        """
        if "pytest" in sys.modules:
            return
        self.logger("\nAvailable cards:")
        cards = list(available_cards.values())
        for i, card in enumerate(cards):
            self.logger(f"{i}: {card}")

    def fill_remaining_slots(
        self, hands: List[List[Card]], available_cards: Dict[str, Card]
    ) -> None:
        """Fill remaining hand slots with random cards.

        Args:
            hands (List[List[Card]]): List of player hands to fill.
            available_cards (Dict[str, Card]): Dictionary of available cards.

        Raises:
            ValueError: If there aren't enough cards left to fill the hands.

        Note:
            - Player 0's hand is filled to 5 cards
            - Player 1's hand is filled to 6 cards
        """
        cards = list(available_cards.values())

        # Fill player 0's hand to 5 cards
        while len(hands[0]) < 5:
            time.sleep(0.1)  # Add small delay to prevent log spam
            if not cards:  # Check if we have any cards left
                raise ValueError("No cards left to fill hands")
            card = random.choice(cards)
            hands[0].append(card)
            del available_cards[str(card)]
            cards.remove(card)
            self.logger(f"Randomly added to Player 0's hand: {card}")

        # Fill player 1's hand to 6 cards
        while len(hands[1]) < 6:
            time.sleep(0.1)  # Add small delay to prevent log spam
            if not cards:  # Check if we have any cards left
                raise ValueError("No cards left to fill hands")
            card = random.choice(cards)
            hands[1].append(card)
            del available_cards[str(card)]
            cards.remove(card)
            self.logger(f"Randomly added to Player 1's hand: {card}")

    def generate_all_cards(self) -> List[Card]:
        """Generate a complete deck of cards without shuffling.

        Returns:
            List[Card]: A list of all possible cards in the game,
                each with a unique UUID.
        """
        cards = []
        for suit in Suit.__members__.values():
            for rank in Rank.__members__.values():
                id = uuid.uuid4()
                cards.append(Card(id, suit, rank))
        return cards

    def generate_shuffled_deck(self) -> List[Card]:
        """Generate a shuffled deck of cards.

        Returns:
            List[Card]: A randomly shuffled complete deck of cards.
        """
        cards = self.generate_all_cards()
        random.shuffle(cards)
        return cards

    def deal_cards(self, deck: List[Card]) -> List[List[Card]]:
        """Deal initial cards to players.

        Args:
            deck (List[Card]): The deck to deal from.

        Returns:
            List[List[Card]]: A list containing two hands:
                - index 0: Player 0's hand (5 cards)
                - index 1: Player 1's hand (6 cards)
        """
        hands = [deck[0:5], deck[5:11]]
        return hands

    def initialize_with_test_deck(self, test_deck: List[Card]) -> None:
        """Initialize the game with a predefined deck for testing.

        Args:
            test_deck (List[Card]): The predefined deck to use.

        Note:
            Deals cards in the same pattern as normal initialization:
            - 5 cards to player 0
            - 6 cards to player 1
            - remaining cards form the deck
        """
        hands = self.deal_cards(test_deck)
        fields = [[], []]
        self.game_state = GameState(hands, fields, test_deck[11:], [])
