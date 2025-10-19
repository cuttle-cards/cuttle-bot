"""
Game state module for the Cuttle card game.

This module provides the GameState class that manages the core game mechanics,
including card playing, turn management, scoring, and win conditions. It handles
all game rules and state transitions.
"""

from __future__ import annotations

from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    cast)

from game.action import Action, ActionType
from game.card import Card, Purpose, Rank
from game.game_history import GameHistory
from game.utils import log_print

# Import AIPlayer only for type checking to avoid circular import
if TYPE_CHECKING:
    from game.ai_player import AIPlayer


class GameState:
    """A class that represents the state of a Cuttle game.

    This class manages all aspects of the game state, including:
    - Player hands and fields
    - Deck and discard pile
    - Turn management
    - Score tracking
    - Game rules enforcement
    - Action resolution
    - Win condition checking

    Attributes:
        hands (List[List[Card]]): The hands of both players. Index 0 is player 0's hand.
        fields (List[List[Card]]): The fields of both players. Index 0 is player 0's field.
        deck (List[Card]): The remaining cards in the deck.
        discard_pile (List[Card]): The cards in the discard pile.
        turn (int): Current player's turn (0 or 1).
        last_action_played_by (Optional[int]): The player who played the last action.
        current_action_player (int): The player currently taking an action.
        status (Optional[str]): Current game status message.
        resolving_two (bool): Whether a Two's effect is being resolved.
        resolving_one_off (bool): Whether a one-off effect is being resolved.
        resolving_three (bool): Whether a Three's effect is being resolved.
        one_off_card_to_counter (Optional[Card]): The one-off card that can be countered.
        logger (callable): Function to use for logging.
        use_ai (bool): Whether AI player is enabled.
        ai_player: The AI player instance if enabled.
        overall_turn (int): The total number of turns played.
        game_history (GameHistory): Chronological record of all game actions.
    """

    use_ai: bool
    ai_player: Optional["AIPlayer"]
    one_off_card_to_counter: Optional[Card] = None
    status: Optional[str] = None
    last_action_played_by: Optional[int] = None

    def __init__(
        self,
        hands: List[List[Card]],
        fields: List[List[Card]],
        deck: List[Card],
        discard_pile: List[Card],
        logger: Callable[..., Any] = print,
        use_ai: bool = False,
        ai_player: Optional["AIPlayer"] = None,
    ):
        """Initialize a new game state.

        Args:
            hands (List[List[Card]]): Initial hands for both players.
                Index 0 is player 0's hand (5 cards).
                Index 1 is player 1's hand (6 cards).
            fields (List[List[Card]]): Initial fields for both players (usually empty).
            deck (List[Card]): Initial deck of cards.
            discard_pile (List[Card]): Initial discard pile (usually empty).
            logger (callable, optional): Function to use for logging. Defaults to print.
            use_ai (bool, optional): Whether to use AI player. Defaults to False.
            ai_player (optional): AI player instance. Defaults to None.
        """
        self.hands = hands
        self.fields = fields
        self.deck = deck
        self.discard_pile = discard_pile
        self.turn = 0  # 0 for p0, 1 for p1
        self.current_action_player = self.turn
        self.status = None
        self.resolving_two = False
        self.resolving_one_off = False
        self.resolving_three = False
        self.one_off_card_to_counter = None
        self.logger = logger
        self.use_ai = use_ai
        self.ai_player = ai_player
        self.overall_turn = 0
        self.last_action_played_by = None
        self.game_history = GameHistory()

    def next_turn(self) -> None:
        """Advance to the next player's turn.

        This method:
        1. Updates the turn counter
        2. Updates the current action player
        3. Increments the overall turn counter if returning to player 0
        4. Increments the game history turn counter
        """
        self.turn = (self.turn + 1) % len(self.hands)
        self.current_action_player = self.turn
        if self.turn == 0:
            self.overall_turn += 1
        self.game_history.increment_turn()

    def next_player(self) -> None:
        """Move to the next player in the action sequence.

        Used during card effect resolution when multiple players
        need to take actions (e.g., countering one-off effects).
        """
        self.current_action_player = (self.current_action_player + 1) % len(self.hands)

    def is_game_over(self) -> bool:
        """Check if the game is over.

        Returns:
            bool: True if there is a winner, False otherwise.
        """
        return self.winner() is not None

    def player_point_cards(self, player: int) -> List[Card]:
        """Get all point cards that count towards a player's score.

        This includes:
        - Point cards on the player's field that haven't been stolen
        - Point cards on the opponent's field that have been stolen by the player

        Args:
            player (int): The player index (0 or 1).

        Returns:
            List[Card]: List of cards that count towards the player's score.
        """
        point_cards = []
        player_field = self.fields[player]
        for card in player_field:
            if card.purpose == Purpose.POINTS and not card.is_stolen():
                point_cards.append(card)
        opponent = (player + 1) % len(self.hands)
        for card in self.fields[opponent]:
            if card.purpose == Purpose.POINTS and card.is_stolen():
                point_cards.append(card)
        return point_cards

    def get_player_score(self, player: int) -> int:
        """Calculate a player's current score.

        Args:
            player (int): The player index (0 or 1).

        Returns:
            int: The sum of all point values from the player's point cards.
        """
        return sum([card.point_value() for card in self.player_point_cards(player)])

    def get_player_field(self, player: int) -> List[Card]:
        """Get all cards that are effectively on a player's field.

        This includes:
        - All non-point cards on their field
        - Point cards on their field that haven't been stolen
        - Point cards on the opponent's field that they've stolen

        Args:
            player (int): The player index (0 or 1).

        Returns:
            List[Card]: List of cards effectively on the player's field.
        """
        field = []
        for card in self.fields[player]:
            if card.purpose != Purpose.POINTS:
                field.append(card)

        for card in self.fields[player]:
            if card.purpose == Purpose.POINTS and not card.is_stolen():
                field.append(card)

        opponent = (player + 1) % len(self.hands)
        for card in self.fields[opponent]:
            if card.purpose == Purpose.POINTS and card.is_stolen():
                field.append(card)
        return field

    def get_player_target(self, player: int) -> int:
        """Calculate a player's current target score based on Kings.

        The target score is determined by the number of Kings on the player's field:
        - 0 Kings: target is 21
        - 1 King:  target is 14
        - 2 Kings: target is 10
        - 3 Kings: target is 5
        - 4 Kings: target is 0

        Args:
            player (int): The player index (0 or 1).

        Returns:
            int: The target score needed to win.
        """
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
        """Check if a player has won the game.

        A player wins if their score equals or exceeds their target score.

        Args:
            player (int): The player index (0 or 1).

        Returns:
            bool: True if the player has won, False otherwise.
        """
        return self.get_player_score(player) >= self.get_player_target(player)

    def winner(self) -> Optional[int]:
        """Get the winning player if the game is over.

        Returns:
            Optional[int]: The winning player's index (0 or 1),
                or None if the game isn't over.
        """
        for player in range(len(self.hands)):
            if self.is_winner(player):
                return player
        return None

    def is_stalemate(self) -> bool:
        """Check if the game has reached a stalemate.

        A stalemate occurs when:
        1. The deck is empty
        2. No player has won

        Returns:
            bool: True if the game is in stalemate, False otherwise.
        """
        # Remove debug print statement that was causing infinite loop
        return self.deck == [] and not self.winner()

    def _move_card_to_discard(self, card: Card) -> None:
        """Move a card to the discard pile along with all its attachments.
        
        Args:
            card (Card): The card to move to the discard pile.
        """
        card.clear_player_info()
        self.discard_pile.append(card)
        for attached_card in card.attachments:
            attached_card.clear_player_info()
            self.discard_pile.append(attached_card)

    def _record_action_to_history(self, action: Action) -> None:
        """Convert an Action to a GameHistoryEntry and record it in game history.
        
        Args:
            action (Action): The action to record in the game history.
        """
        # Determine source and destination based on action type
        source = ""
        destination = ""
        
        if action.action_type == ActionType.DRAW:
            source = "deck"
            destination = "hand"
        elif action.action_type in [ActionType.POINTS, ActionType.FACE_CARD]:
            source = "hand"
            destination = "field"
        elif action.action_type in [ActionType.ONE_OFF, ActionType.SCUTTLE]:
            source = "hand"
            destination = "discard_pile"
        elif action.action_type == ActionType.COUNTER:
            source = "hand"
            destination = "discard_pile"
        
        # Record the action in game history
        self.game_history.record_action(
            player=action.played_by,
            action_type=action.action_type,
            card=action.card,
            target=action.target,
            source=source,
            destination=destination,
        )

    def update_state(self, action: Action) -> Tuple[bool, bool, Optional[int]]:
        """Update the game state based on an action.

        This is the main method for executing game actions. It handles:
        - Drawing cards
        - Playing point cards
        - Scuttling cards
        - One-off effects
        - Countering
        - Effect resolution

        Args:
            action (Action): The action to execute.

        Returns:
            Tuple[bool, bool, Optional[int]]: A tuple containing:
                - bool: Whether the turn is finished
                - bool: Whether the game should stop
                - Optional[int]: The winner's index if game is over, None otherwise
        """
        # Record the action in game history first
        self._record_action_to_history(action)
        
        turn_finished = False
        should_stop = False
        winner = None

        if action.action_type == ActionType.DRAW:
            self.draw_card()
            turn_finished = True
            return turn_finished, should_stop, winner
        elif action.action_type == ActionType.POINTS:
            if action.card is not None:
                won = self.play_points(action.card)
                turn_finished = True
                if won:
                    should_stop = True
                    winner = self.winner()
                    return turn_finished, should_stop, winner
            else:
                # Handle error: POINTS action requires a card
                log_print("Error: POINTS action called without a card.")
                return True, True, None # Stop game on error
        elif action.action_type == ActionType.SCUTTLE:
            if action.card is not None and action.target is not None:
                self.scuttle(action.card, action.target)
                turn_finished = True
                should_stop = False  # scuttle doesn't end the game
                winner = self.winner()
                return turn_finished, should_stop, winner
            else:
                # Handle error: SCUTTLE action requires card and target
                log_print("Error: SCUTTLE action called without card or target.")
                return True, True, None # Stop game on error
        elif action.action_type == ActionType.ONE_OFF:
            if action.card is not None:
                # Normal one-off handling for all cards
                turn_finished, played_by = self.play_one_off(
                    self.turn, action.card, None, None
                )
                if turn_finished:
                    winner = self.winner()
                    should_stop = winner is not None
                    return turn_finished, should_stop, winner
                self.resolving_one_off = True
                self.one_off_card_to_counter = action.card
                return turn_finished, should_stop, winner
            else:
                # Handle error: ONE_OFF action requires a card
                log_print("Error: ONE_OFF action called without a card.")
                return True, True, None # Stop game on error
        elif action.action_type == ActionType.COUNTER:
            if action.card is not None and action.target is not None:
                action.card.purpose = Purpose.COUNTER
                if action.card.played_by is not None: # Check played_by before use
                    self.current_action_player = action.card.played_by
                else:
                    self.current_action_player = action.played_by
                    action.card.played_by = action.played_by
                    log_print(f"Counter card {action.card} has no played_by")

                turn_finished, played_by = self.play_one_off(
                    player=self.turn,
                    card=action.target, # Target is the card being countered
                    countered_with=action.card, # Card is the Two used to counter
                    last_resolved_by=None,
                )
                if turn_finished:
                    winner = self.winner()
                    should_stop = winner is not None
                    return turn_finished, should_stop, winner
            else:
                # Handle error: COUNTER action requires card and target
                log_print("Error: COUNTER action called without card or target.")
                return True, True, None # Stop game on error
        elif action.action_type == ActionType.RESOLVE:
            if action.target is not None:
                turn_finished, played_by = self.play_one_off(
                    self.turn, action.target, None, action.played_by
                )
                if turn_finished:
                    winner = self.winner()
                    should_stop = winner is not None
                    return turn_finished, should_stop, winner
            else:
                # Handle error: RESOLVE action requires a target
                log_print("Error: RESOLVE action called without a target.")
                return True, True, None # Stop game on error
        elif action.action_type == ActionType.FACE_CARD:
            if action.card is not None:
                won = self.play_face_card(action.card, action.target) # Target can be None for King/Queen
                turn_finished = True
                if won:
                    should_stop = True
                    winner = self.turn
                return turn_finished, should_stop, winner
            else:
                # Handle error: FACE_CARD action requires a card
                log_print("Error: FACE_CARD action called without a card.")
                return True, True, None # Stop game on error
        elif action.action_type == ActionType.JACK:
            if action.card is not None and action.target is not None:
                # Check if opponent has a queen on their field
                # implement play_face_card with optional target
                won = self.play_face_card(action.card, action.target)
                turn_finished = True
                if won:
                    should_stop = True
                    winner = self.turn
                return turn_finished, should_stop, winner
            else:
                # Handle error: JACK action requires card and target
                log_print("Error: JACK action called without card or target.")
                return True, True, None # Stop game on error

        return turn_finished, should_stop, winner

    def draw_card(self, count: int = 1) -> None:
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
            card = self.deck.pop()
            self.hands[self.turn].append(card)
            # Record each individual card draw (for multi-card draws like 5s)
            if count > 1:
                self.game_history.record_action(
                    player=self.turn,
                    action_type=ActionType.DRAW,
                    card=card,
                    source="deck",
                    destination="hand",
                )

    def play_points(self, card: Card) -> bool:
        # play a points card
        self.hands[self.turn].remove(card)
        card.purpose = Purpose.POINTS
        card.played_by = self.turn
        self.fields[self.turn].append(card)

        # check if the player has won
        if self.get_player_score(self.turn) >= self.get_player_target(self.turn):
            print(
                f"Player {self.turn} wins! Score: {self.get_player_score(self.turn)} points (target: {self.get_player_target(self.turn)} with {len([c for c in self.fields[self.turn] if c.rank == Rank.KING])} Kings)"
            )
            self.status = "win"
            return True
        return False

    def scuttle(self, card: Card, target: Card) -> None:
        # Validate scuttle conditions
        if (
            card.point_value() == target.point_value()
            and card.suit_value() <= target.suit_value()
        ):
            raise Exception(
                "Invalid scuttle: Cannot scuttle with a lower or equal suit of the same rank"
            )

        # scuttle a points card
        card.played_by = self.turn
        card_player = card.played_by
        if card_player is not None:
            if card in self.hands[card_player]:
                log_print(f"Removing card {card} from card player's hand")
                self.hands[card_player].remove(card)
            else:
                log_print(f"Card {card} not found on card player's hand")
                raise Exception(f"Card {card} not found on card player's hand")
        self._move_card_to_discard(card)

        target_player = target.played_by
        if target_player is not None:
            if target in self.fields[target_player]:
                log_print(f"Removing target card {target} from target player's field")
                self.fields[target_player].remove(target)
            else:
                log_print(f"Target card {target} not found on target player's field")
                raise Exception(f"Target card {target} not found on target player's field")
        self._move_card_to_discard(target)

    def play_one_off(
        self,
        player: int,
        card: Card,
        countered_with: Optional[Card] = None,
        last_resolved_by: Optional[int] = None,
    ) -> Tuple[bool, Optional[int]]:
        """
        Play a one-off card.

        Args:
            player: The player playing the card
            card: The card being played
            countered_with: The TWO being used to counter, if any
            last_resolved_by: The player who last resolved (didn't counter), if any

        Returns:
            Tuple[bool, int | None]:
                - Whether the turn is finished
                - The player who played the last action (for tracking counter chain)
        """
        # Initial play requires additional input
        self.last_action_played_by = player
        if player == self.turn and countered_with is None and last_resolved_by is None:
            # Initial play of one-off, waiting for counter/resolve
            return False, None

        # Handle counter with a Two
        if countered_with is not None:
            # Validate counter card
            if countered_with.point_value() != 2:
                raise Exception("Counter must be a 2")
            if countered_with.purpose != Purpose.COUNTER:
                raise Exception(
                    f"Counter must be with a purpose of counter, instead got {countered_with.purpose}"
                )
            counter_player = countered_with.played_by
            log_print(f"Checking queen on opponent's field for counter card {countered_with}")
            if counter_player is not None:
                other_player = (counter_player + 1) % len(self.hands)
                # check if other player has a queen on their field
                other_player_field = self.fields[other_player]
                queen_on_opponent_field = any(
                    card.rank == Rank.QUEEN for card in other_player_field
                )
                if queen_on_opponent_field:
                    raise Exception(
                        "Cannot counter with a two if opponent has a queen on their field"
                    )

            # Move counter card to discard pile
            log_print(f"Moving counter card {countered_with} to discard pile")
            played_by = countered_with.played_by
            if played_by is not None and countered_with in self.hands[played_by]:
                self.hands[played_by].remove(countered_with)
                self._move_card_to_discard(countered_with)
                log_print(f"Counter card {countered_with} moved to discard pile")

            # Move the countered card to discard pile if it's still in hand
            if card in self.hands[self.turn]:
                self.hands[self.turn].remove(card)
            if card not in self.discard_pile:
                self._move_card_to_discard(card)

            # Update last action for counter chain
            self.last_action_played_by = played_by
            return False, played_by

        # Handle resolution (no counter)
        if last_resolved_by is not None:
            self.last_action_played_by = player

            if last_resolved_by != self.turn:
                # Opponent didn't counter, so one-off resolves
                if card in self.hands[self.turn]:
                    self.hands[self.turn].remove(card)
                card.purpose = Purpose.ONE_OFF
                self.apply_one_off_effect(card)
                if card not in self.discard_pile:
                    self._move_card_to_discard(card)
            else:
                # Original player accepts counter
                # One-off is countered, move to discard
                if card in self.hands[self.turn]:
                    self.hands[self.turn].remove(card)
                if card not in self.discard_pile:
                    self._move_card_to_discard(card)

            # Turn is finished after resolution
            return True, None

        return True, None

    def apply_one_off_effect(self, card: Card) -> None:
        print(f"Applying one off effect for {card}")
        print(len(self.hands[self.turn]))
        if card.rank == Rank.ACE:
            # Clear all point cards from all players' fields
            for player_field in self.fields:
                point_cards = [
                    card
                    for card in player_field
                    if card.is_point_card() and card.purpose == Purpose.POINTS
                ]
                for point_card in point_cards:
                    player_field.remove(point_card)
                    self._move_card_to_discard(point_card)
        elif card.rank == Rank.THREE:
            # Allow player to take a card from the discard pile
            if not self.discard_pile:
                print("No cards in discard pile to take")
                return

            # Get the player's choice
            print(f"self.use_ai: {self.use_ai}")
            print(f"self.turn: {self.turn}")
            chosen_card = None
            if self.use_ai and self.turn == 1:
                if self.ai_player is not None:
                    chosen_card = self.ai_player.choose_card_from_discard(self.discard_pile)
                    if chosen_card in self.discard_pile:
                        self.discard_pile.remove(chosen_card)
                    self.hands[self.turn].append(chosen_card)
                    print(f"AI chose {chosen_card} from discard pile")
                else:
                    print("Warning: AI player is None, cannot choose card.")
                    if self.discard_pile:
                        chosen_card = self.discard_pile.pop(0)
                        self.hands[self.turn].append(chosen_card)
            else:
                # Create a list of card options for the input handler
                card_options = [str(card) for card in self.discard_pile]

                # Use the input handler to get the player's choice
                from game.input_handler import get_interactive_input

                index = get_interactive_input(
                    "Select a card from the discard pile:", card_options
                )

                # Handle the selection
                if 0 <= index < len(self.discard_pile):
                    chosen_card = self.discard_pile.pop(index)
                    chosen_card.clear_player_info()
                    self.hands[self.turn].append(chosen_card)
                    print(f"Took {chosen_card} from discard pile")
                else:
                    print("Invalid selection")
        elif card.rank == Rank.FOUR:
            # Opponent needs to select 2 cards from their hand to discard
            # if opponent only has 1 card, they can discard that one

            # Get the player's choice
            print(f"self.use_ai: {self.use_ai}")
            print(f"self.turn: {self.turn}")
            chosen_cards = None
            opponent = (self.turn + 1) % len(self.hands)
            discard_prompt = f"player {opponent} must discard 2 cards"
            if len(self.hands[opponent]) == 1:
                discard_prompt = f"player {opponent} must discard 1 card"
            if len(self.hands[opponent]) == 0:
                discard_prompt = f"player {opponent} has no cards to discard"
                log_print(discard_prompt)
                # end turn
                return
            log_print(discard_prompt)

            if self.use_ai and self.current_action_player == opponent:
                if self.ai_player is not None:
                    chosen_cards = self.ai_player.choose_two_cards_from_hand(
                        self.hands[opponent]
                    )
                    log_print(f"AI chose {chosen_cards} from hand to discard")
                    for chosen_card in chosen_cards:
                        if chosen_card in self.hands[opponent]:
                            self.hands[opponent].remove(chosen_card)
                            self.discard_pile.append(chosen_card)
                            chosen_card.clear_player_info()
                else:
                    print("Warning: AI player is None, cannot choose cards.")
                    num_to_discard = min(2, len(self.hands[opponent]))
                    for _ in range(num_to_discard):
                        if self.hands[opponent]:
                           discarded_card = self.hands[opponent].pop(0)
                           self.discard_pile.append(discarded_card)
                           discarded_card.clear_player_info()
            else:
                cards_to_discard = []
                cards_remaining = self.hands[opponent].copy()

                # Determine how many cards to discard
                num_cards_to_discard = min(2, len(cards_remaining))

                for i in range(num_cards_to_discard):
                    if not cards_remaining:
                        break

                    # Create a list of card options for the input handler
                    card_options = [str(card) for card in cards_remaining]

                    # Use the input handler to get the player's choice
                    from game.input_handler import get_interactive_input

                    index = get_interactive_input(
                        f"Select card {i + 1} to discard:", card_options
                    )

                    # Handle the selection
                    if 0 <= index < len(cards_remaining):
                        chosen_card = cards_remaining.pop(index)
                        cards_to_discard.append(chosen_card)
                        log_print(f"Opponent discarded {chosen_card}")

                        # Remove card from opponent's hand and add to discard pile
                        self.hands[opponent].remove(chosen_card)
                        self.discard_pile.append(chosen_card)
                        chosen_card.clear_player_info()
                    else:
                        log_print("Invalid selection")
        elif card.rank == Rank.FIVE:
            if len(self.hands[self.turn]) <= 6:
                self.draw_card(2)
            elif len(self.hands[self.turn]) == 7:
                self.draw_card(1)
            else:
                pass
        elif card.rank == Rank.SIX:
            # Clear all face cards from all players' fields
            for player_field in self.fields:
                face_cards = [
                    card
                    for card in player_field
                    if card.is_face_card() and card.purpose == Purpose.FACE_CARD
                ]
                for face_card in face_cards:
                    player_field.remove(face_card)
                    self._move_card_to_discard(face_card)

    def play_face_card(self, card: Card, target: Optional[Card] = None) -> bool:
        """Play a face card (King, Queen, Jack) from hand to field.

        Args:
            card (Card): The face card to play
            target (Card, optional): The target card for Jack. Required for Jack, ignored for other face cards.

        Returns:
            bool: True if the player has won, False otherwise
        """
        # For Jack, target is required and must be a point card
        if card.rank == Rank.JACK:
            if target is None:
                raise Exception("Target card is required for playing Jack")
            if target.purpose != Purpose.POINTS: # Check purpose after confirming target is not None
                raise Exception("Target card must be a point card for playing Jack")

        # Remove from hand and add to field/attachments
        if card.rank != Rank.JACK:
            if card not in self.hands[self.turn]:
                raise Exception(f"Can only play cards from your hand, card: {card} not in hand: {self.hands[self.turn]}")
            self.hands[self.turn].remove(card)
            card.purpose = Purpose.FACE_CARD
            card.played_by = self.turn
            self.fields[self.turn].append(card)

            # Check for instant win with King (if points already meet new target)
            if card.rank == Rank.KING and self.is_winner(self.turn):
                print(
                    f"Player {self.turn} wins! Score: {self.get_player_score(self.turn)} points (target: {self.get_player_target(self.turn)} with {len([c for c in self.fields[self.turn] if c.rank == Rank.KING])} Kings)"
                )
                self.status = "win"
                return True

            return False # Return False if not King win
        else: # Handling Jack
            target = cast(Card, target)
            opponent = (self.turn + 1) % len(self.hands)
            queen_on_opponent_field = any(
                c.rank == Rank.QUEEN for c in self.fields[opponent]
            )
            if queen_on_opponent_field:
                raise Exception(
                    "Cannot play jack as face card if opponent has a queen on their field"
                )

            # Target is guaranteed not None here due to earlier check
            # Verify target is a point card (redundant check, but safe)
            if not target.is_point_card() or target.purpose != Purpose.POINTS:
                raise Exception("Jack can only be played on point cards")

            # Remove Jack from hand
            if card not in self.hands[self.turn]:
                 raise Exception(f"Can only play cards from your hand, card: {card} not in hand: {self.hands[self.turn]}")
            self.hands[self.turn].remove(card)
            card.purpose = Purpose.JACK
            card.played_by = self.turn

            # Attach Jack to the target card
            target.attachments.append(card) # target confirmed not None

            if self.winner() is not None:
                return True
        return False

    def get_legal_actions(self) -> List[Action]:
        """
        Get all legal actions for the current player.

        Returns:
            List[Action]: A list of legal actions
        """
        actions = []

        # If resolving three, THIS IS HANDLED BY apply_one_off_effect
        # No specific actions needed here, the user/AI interaction happens there
        if self.resolving_three:
            # Returning empty list or specific instruction might be better
            # For now, let's assume apply_one_off_effect handles the choice
            # Or perhaps we need an ActionType.CHOOSE_FROM_DISCARD?
            # For mypy, let's just bypass this section for action generation
            return [] # Or handle as appropriate for game flow

        # If resolving one-off, only allow counter or resolve
        if self.resolving_one_off:
            # Check if current player has a Two to counter with
            twos = [
                card
                for card in self.hands[self.current_action_player]
                if card.rank == Rank.TWO
            ]
            # if opponent has a queen on their field, can't counter with a two, cannot counter
            other_player = (self.current_action_player + 1) % len(self.hands)
            other_player_field = self.fields[other_player]
            queen_on_opponent_field = any(
                card.rank == Rank.QUEEN for card in other_player_field
            )

            if queen_on_opponent_field:
                log_print(
                    "Cannot counter with a two if opponent has a queen on their field"
                )
            else:
                for two in twos:
                    actions.append(
                        Action(
                            ActionType.COUNTER,
                            self.current_action_player,
                            card=two,
                            target=self.one_off_card_to_counter,
                        )
                    )
            # Always allow resolving (not countering)
            actions.append(
                Action(
                    ActionType.RESOLVE,
                    self.current_action_player,
                    target=self.one_off_card_to_counter,
                )
            )
            return actions

        # Allow drawing a card only if hand is not full (max 8 cards)
        if len(self.hands[self.turn]) < 8:
            actions.append(Action(ActionType.DRAW, self.turn))

        # Get cards in current player's hand
        hand = self.hands[self.turn]

        # Can play any card as points (2-10)
        for card in hand:
            if card.point_value() <= Rank.TEN.value[1]:
                actions.append(Action(ActionType.POINTS, self.turn, card=card))

        # Can play face cards
        for card in hand:
            # Only Kings are implemented for now
            # TODO: Implement Queens, Jacks, and Eights
            if card.is_face_card() and card.rank in [Rank.KING, Rank.QUEEN]:
                actions.append(Action(ActionType.FACE_CARD, self.turn, card=card))

        opponent = (self.current_action_player + 1) % len(self.hands)
        queen_on_opponent_field = any(
            card.rank == Rank.QUEEN for card in self.fields[opponent]
        )
        # Can play Jacks on opponent's point cards on field
        for card in hand:
            if card.rank == Rank.JACK and not queen_on_opponent_field:
                for opponent_card in self.get_player_field(opponent):
                    if opponent_card.purpose == Purpose.POINTS:
                        # TODO: also check if card has jacks attached
                        actions.append(
                            Action(ActionType.JACK, self.turn, card=card, target=opponent_card)
                        )

        # Can play one-offs
        for card in hand:
            if card.is_one_off():
                actions.append(Action(ActionType.ONE_OFF, self.turn, card=card))

        # Can scuttle opponent's point cards with higher point cards (only point cards can scuttle)
        opponent = (self.turn + 1) % len(self.hands)
        opponent_field = self.fields[opponent]
        opponent_points = [
            card for card in opponent_field if card.purpose == Purpose.POINTS
        ]

        # Get point cards from hand (Ace to Ten)
        point_cards = [card for card in hand if card.point_value() <= Rank.TEN.value[1]]

        print(f"opponent_points: {opponent_points}")
        # For each point card in opponent's field
        for opponent_card in opponent_points:
            # For each point card in player's hand
            for card in point_cards:
                # Can scuttle if:
                # 1. Higher point value, or
                # 2. Equal point value and higher suit value
                if card.point_value() > opponent_card.point_value() or (
                    card.point_value() == opponent_card.point_value()
                    and card.suit_value() > opponent_card.suit_value()
                ):
                    actions.append(
                        Action(ActionType.SCUTTLE, self.turn, card=card, target=opponent_card)
                    )
        return actions

    def print_state(self, hide_player_hand: Optional[int] = None) -> None:
        """Print the current game state to the console.

        Args:
            hide_player_hand (Optional[int], optional): Index of the player whose hand
                should be hidden (e.g., for privacy). Defaults to None (show both).
        """
        winner = self.winner()
        if winner is not None:
            self.logger(f"Player {winner} wins!")
            return

        if self.is_stalemate():
            self.logger("Stalemate!")
            return

        self.logger("\n" + "=" * 20)
        self.logger(f"Turn: Player {self.turn} (Overall Turn: {self.overall_turn})")
        self.logger(f"Current Action Player: {self.current_action_player}")

        for player in range(len(self.hands)):
            self.logger("-" * 20)
            self.logger(
                f"Player {player}: Score = {self.get_player_score(player)}, Target = {self.get_player_target(player)}"
            )
            # Use a check for None before comparing hide_player_hand
            if hide_player_hand is not None and hide_player_hand == player:
                self.logger(f"  Hand: [{len(self.hands[player])} cards hidden]")
            else:
                hand_str = ", ".join(map(str, self.hands[player]))
                self.logger(f"  Hand: [{hand_str}]")

            field_str = ", ".join(map(str, self.get_player_field(player)))
            self.logger(f"  Field: [{field_str}]")

        self.logger("-" * 20)
        self.logger(f"Deck: {len(self.deck)} cards remaining")
        discard_str = ", ".join(map(str, self.discard_pile))
        self.logger(f"Discard Pile: [{discard_str}]")

        if self.status:
            self.logger(f"Status: {self.status}")
        self.logger("=" * 20 + "\n")

    def to_dict(self) -> Dict:
        """
        Convert the game state to a dictionary for saving.

        Returns:
            A dictionary representation of the game state
        """
        return {
            "hands": [[card.to_dict() for card in hand] for hand in self.hands],
            "fields": [[card.to_dict() for card in field] for field in self.fields],
            "deck": [card.to_dict() for card in self.deck],
            "discard_pile": [card.to_dict() for card in self.discard_pile],
            "turn": self.turn,
            "last_action_played_by": self.last_action_played_by,
            "current_action_player": self.current_action_player,
            "status": self.status,
            "resolving_two": self.resolving_two,
            "resolving_one_off": self.resolving_one_off,
            "resolving_three": self.resolving_three,
            "one_off_card_to_counter": self.one_off_card_to_counter.to_dict()
            if self.one_off_card_to_counter is not None
            else None,
            "use_ai": self.use_ai,
            "overall_turn": self.overall_turn,
            "game_history": self.game_history.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict, logger: Callable[..., Any] = print) -> "GameState":
        """
        Create a game state from a dictionary.

        Args:
            data: A dictionary representation of a game state

        Returns:
            A new GameState instance
        """
        hands_data = data.get("hands", [[], []])
        fields_data = data.get("fields", [[], []])
        deck_data = data.get("deck", [])
        discard_pile_data = data.get("discard_pile", [])
        one_off_counter_data = data.get("one_off_card_to_counter")

        hands = [[Card.from_dict(card) for card in hand] for hand in hands_data]
        fields = [[Card.from_dict(card) for card in field] for field in fields_data]
        deck = [Card.from_dict(card) for card in deck_data]
        discard_pile = [Card.from_dict(card) for card in discard_pile_data]

        state = cls(
            hands=hands,
            fields=fields,
            deck=deck,
            discard_pile=discard_pile,
            logger=logger,
            use_ai=data.get("use_ai", False),
        )
        state.turn = data.get("turn", 0)
        state.last_action_played_by = data.get("last_action_played_by")
        state.current_action_player = data.get("current_action_player", state.turn)
        state.status = data.get("status")
        state.resolving_two = data.get("resolving_two", False)
        state.resolving_one_off = data.get("resolving_one_off", False)
        state.resolving_three = data.get("resolving_three", False)
        state.one_off_card_to_counter = (
            Card.from_dict(one_off_counter_data)
            if one_off_counter_data is not None
            else None
        )
        state.ai_player = None  # Placeholder, actual instance set by Game
        state.overall_turn = data.get("overall_turn", 0)
        
        # Load game history if present, otherwise create new empty history
        history_data = data.get("game_history")
        if history_data:
            state.game_history = GameHistory.from_dict(history_data)
        else:
            state.game_history = GameHistory()

        return state
