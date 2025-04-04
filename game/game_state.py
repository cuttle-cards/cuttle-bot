from __future__ import annotations

from typing import List, Optional, Dict
from game.card import Card, Purpose, Rank
from game.action import Action, ActionType, ActionSource
from game.utils import log_print


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
        logger: callable - The logger to use for output.

    """

    use_ai: bool
    ai_player: None

    def __init__(
        self,
        hands: List[List[Card]],
        fields: List[List[Card]],
        deck: List[Card],
        discard_pile: List[Card],
        logger=print,  # Default to print if no logger provided
        use_ai: bool = False,
        ai_player=None,
    ):
        """
        Initialize the game state.
        """
        self.hands = hands
        self.fields = fields
        self.deck = deck
        self.discard_pile = discard_pile
        self.turn = 0  # 0 for p0, 1 for p1
        self.last_action_played_by = None
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

    def next_turn(self):
        self.turn = (self.turn + 1) % len(self.hands)
        self.current_action_player = self.turn
        if self.turn == 0:
            self.overall_turn += 1

    def next_player(self):
        self.current_action_player = (self.current_action_player + 1) % len(self.hands)

    def is_game_over(self) -> bool:
        return self.winner() is not None

    def get_player_score(self, player: int) -> int:
        hand = self.hands[player]
        field = self.fields[player]
        point_cards = [
            card
            for card in field
            if card.point_value() <= Rank.TEN.value[1]
            and card.purpose == Purpose.POINTS
        ]

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
            turn_finished = True
            should_stop = False  # scuttle doesn't end the game
            winner = self.winner()
            return turn_finished, should_stop, winner
        elif action.action_type == ActionType.ONE_OFF:
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
        elif action.action_type == ActionType.COUNTER:
            action.card.purpose = Purpose.COUNTER
            action.card.played_by = self.current_action_player
            turn_finished, played_by = self.play_one_off(
                player=self.turn,
                card=action.target,
                countered_with=action.card,
                last_resolved_by=None,
            )
            if turn_finished:
                winner = self.winner()
                should_stop = winner is not None
                return turn_finished, should_stop, winner
        elif action.action_type == ActionType.RESOLVE:
            turn_finished, played_by = self.play_one_off(
                self.turn, action.target, None, action.played_by
            )
            if turn_finished:
                winner = self.winner()
                should_stop = winner is not None
                return turn_finished, should_stop, winner
        elif action.action_type == ActionType.FACE_CARD:
            won = self.play_face_card(action.card)
            turn_finished = True
            if won:
                should_stop = True
                winner = self.turn
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
            print(
                f"Player {self.turn} wins! Score: {self.get_player_score(self.turn)} points (target: {self.get_player_target(self.turn)} with {len([c for c in self.fields[self.turn] if c.rank == Rank.KING])} Kings)"
            )
            self.status = "win"
            return True
        return False

    def scuttle(self, card: Card, target: Card):
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
        self.hands[card.played_by].remove(card)
        card.clear_player_info()
        self.discard_pile.append(card)
        self.fields[target.played_by].remove(target)
        target.clear_player_info()
        self.discard_pile.append(target)

    def play_one_off(
        self,
        player: int,
        card: Card,
        countered_with: Card = None,
        last_resolved_by: int = None,
    ):
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
            other_player = (countered_with.played_by + 1) % len(self.hands)
            # check if other player has a queen on their field
            other_player_field = self.fields[other_player]
            queen_on_opponent_field = any(card.rank == Rank.QUEEN for card in other_player_field)
            if queen_on_opponent_field:
                raise Exception("Cannot counter with a two if opponent has a queen on their field")

            # Move counter card to discard pile
            played_by = countered_with.played_by
            print(f"played_by: {played_by}")
            print(f"self.hands[played_by]: {self.hands[played_by]}")
            print(f"countered_with: {countered_with}")
            print(
                f"countered_with in self.hands[played_by]: {countered_with in self.hands[played_by]}"
            )
            if countered_with in self.hands[played_by]:
                self.hands[played_by].remove(countered_with)
                self.discard_pile.append(countered_with)
                countered_with.clear_player_info()

            # Move the countered card to discard pile if it's still in hand
            if card in self.hands[self.turn]:
                self.hands[self.turn].remove(card)
            if card not in self.discard_pile:
                self.discard_pile.append(card)
                card.clear_player_info()

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
                card.clear_player_info()
                if card not in self.discard_pile:
                    self.discard_pile.append(card)
            else:
                # Original player accepts counter
                # One-off is countered, move to discard
                if card in self.hands[self.turn]:
                    self.hands[self.turn].remove(card)
                if card not in self.discard_pile:
                    self.discard_pile.append(card)
                card.clear_player_info()

            # Turn is finished after resolution
            return True, None

        return True, None

    def apply_one_off_effect(self, card: Card):
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
                    point_card.clear_player_info()
                    self.discard_pile.append(point_card)
        elif card.rank == Rank.THREE:
            # Allow player to take a card from the discard pile
            if not self.discard_pile:
                print("No cards in discard pile to take")
                return

            # Show available cards in discard pile
            print("\nAvailable cards in discard pile:")
            for i, card in enumerate(self.discard_pile):
                print(f"{i}: {card}")

            # Get the player's choice
            print(f"self.use_ai: {self.use_ai}")
            print(f"self.turn: {self.turn}")
            chosen_card = None
            if self.use_ai and self.turn == 1:  # AI's turn
                # Let AI choose a card
                chosen_card = self.ai_player.choose_card_from_discard(self.discard_pile)
                self.hands[self.turn].append(chosen_card)

                print(f"AI chose {chosen_card} from discard pile")
            else:  # Human player's turn
                while True:
                    try:
                        choice = input("Enter the number of the card to take: ")
                        index = int(choice)
                        if 0 <= index < len(self.discard_pile):
                            chosen_card = self.discard_pile.pop(index)
                            chosen_card.clear_player_info()
                            self.hands[self.turn].append(chosen_card)
                            print(f"Took {chosen_card} from discard pile")
                            break
                        print("Invalid number, please try again")
                    except ValueError:
                        print("Please enter a valid number")
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
            # print the opponent's hand with index
            log_print(f"opponent's hand:")
            for i, card in enumerate(self.hands[opponent]):
                log_print(f"{i}: {card}")
            if self.use_ai and self.current_action_player == opponent:  # AI's turn
                # Let AI choose a card
                chosen_cards = self.ai_player.choose_two_cards_from_hand(self.hands[opponent])
                for card in chosen_cards:
                    self.hands[opponent].remove(card)
                    self.discard_pile.append(card)
                    card.clear_player_info()
            else:  # Human player's turn
                cards_to_discard = []
                while len(cards_to_discard) < 2 and len(self.hands[opponent]) > 0:
                    try:
                        choice = input("Enter the number of the card to discard: ")
                        index = int(choice)
                        if 0 <= index < len(self.hands[opponent]):
                            # remove card from opponent's hand
                            chosen_card = self.hands[opponent].pop(index)
                            cards_to_discard.append(chosen_card)
                            log_print(f"Opponent discarded {chosen_card}")
                            # add card to discard pile
                            self.discard_pile.append(chosen_card)
                            chosen_card.clear_player_info()
                    except ValueError:
                        print("Please enter a valid number")
                # Remove the chosen cards from the player's hand
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
                    face_card.clear_player_info()
                    self.discard_pile.append(face_card)


    def play_face_card(self, card: Card) -> bool:
        """
        Play a face card (King, Queen, Jack, Eight) to the field.

        Args:
            card: The face card to play

        Returns:
            bool: True if this play results in a win, False otherwise

        Raises:
            Exception: If the card is not in the current player's hand
            Exception: If the card is not a face card
        """
        # Validate card is in current player's hand
        if card not in self.hands[self.turn]:
            raise Exception("Can only play cards from your hand")

        # Validate card is a face card
        if not card.is_face_card():
            raise Exception(f"{card} is not a face card")

        # Remove from hand and add to field
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

        return False

    def get_legal_actions(self) -> List[Action]:
        """
        Get all legal actions for the current player.
        """
        actions = []

        # If resolving three, legal actions is to choose a card from the discard pile
        if self.resolving_three:
            for card in self.discard_pile:
                actions.append(Action(ActionType.THREE, card, None, self.turn))
            return actions

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
            queen_on_opponent_field = any(card.rank == Rank.QUEEN for card in other_player_field)

            if queen_on_opponent_field:
                log_print("Cannot counter with a two if opponent has a queen on their field")
            else:
                for two in twos:
                    actions.append(
                        Action(
                            ActionType.COUNTER,
                            two,
                            self.one_off_card_to_counter,
                            self.current_action_player,
                        )
                    )
            # Always allow resolving (not countering)
            actions.append(
                Action(
                    ActionType.RESOLVE,
                    None,
                    self.one_off_card_to_counter,
                    self.current_action_player,
                )
            )
            return actions

        # Always allow drawing a card
        actions.append(Action(ActionType.DRAW, None, None, self.turn))

        # Get cards in current player's hand
        hand = self.hands[self.turn]

        # Can play any card as points (2-10)
        for card in hand:
            if card.point_value() <= Rank.TEN.value[1]:
                actions.append(Action(ActionType.POINTS, card, None, self.turn))

        # Can play face cards
        for card in hand:
            # Only Kings are implemented for now
            # TODO: Implement Queens, Jacks, and Eights
            if card.is_face_card() and card.rank in [Rank.KING, Rank.QUEEN]:
                actions.append(Action(ActionType.FACE_CARD, card, None, self.turn))

        # Can play one-offs
        for card in hand:
            if card.is_one_off():
                actions.append(Action(ActionType.ONE_OFF, card, None, self.turn))

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
                        Action(ActionType.SCUTTLE, card, opponent_card, self.turn)
                    )
        return actions

    def print_state(self, hide_player_hand: int = None):
        print("--------------------------------")
        print(f"Player {self.current_action_player}'s turn")
        print(f"Deck: {len(self.deck)}")
        print(f"Discard Pile: {len(self.discard_pile)}")
        print("Points: ")
        for i, hand in enumerate(self.hands):
            points = self.get_player_score(i)
            print(f"Player {i}: {points}")
        for i, hand in enumerate(self.hands):
            if i == hide_player_hand:
                print(f"Player {i}'s hand: [Hidden]")
            else:
                print(f"Player {i}'s hand: {hand}")
        for i, field in enumerate(self.fields):
            print(f"Player {i}'s field: {field}")
        print("--------------------------------")

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
            "discard": [card.to_dict() for card in self.discard_pile],
            "turn": self.turn,
            "status": self.status,
            "resolving_two": self.resolving_two,
            "resolving_one_off": self.resolving_one_off,
            "one_off_card_to_counter": (
                self.one_off_card_to_counter.to_dict()
                if self.one_off_card_to_counter
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict, logger=print) -> "GameState":
        """
        Create a game state from a dictionary.

        Args:
            data: A dictionary representation of a game state

        Returns:
            A new GameState instance
        """
        hands = [
            [Card.from_dict(card_data) for card_data in hand] for hand in data["hands"]
        ]
        fields = [
            [Card.from_dict(card_data) for card_data in field]
            for field in data["fields"]
        ]
        deck = [Card.from_dict(card_data) for card_data in data["deck"]]
        discard_pile = [Card.from_dict(card_data) for card_data in data["discard"]]

        game_state = cls(hands, fields, deck, discard_pile, logger=logger)
        game_state.turn = data["turn"]
        game_state.status = data["status"]
        game_state.resolving_two = data["resolving_two"]
        game_state.resolving_one_off = data["resolving_one_off"]
        game_state.one_off_card_to_counter = (
            Card.from_dict(data["one_off_card_to_counter"])
            if data["one_off_card_to_counter"]
            else None
        )

        return game_state
