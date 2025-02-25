# Copyright 2025
# Based on the MIT-licensed Durak implementation by Manuel Boesl and Borislav Radev
# https://github.com/ManuelBoesl/durak_card_game

import enum
import numpy as np
import pyspiel
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------------------
# Global definitions and constants
# ----------------------------------------------------------------------------------

_NUM_PLAYERS = 2
_NUM_CARDS = 36  # Cards 0..35; suit = card // 9, rank = card % 9
_CARDS_PER_PLAYER = 6  # Each player is dealt (up to) 6 cards
_DECK = list(range(_NUM_CARDS))


def suit_of(card: int) -> int:
    return card // 9

def rank_of(card: int) -> int:
    return card % 9

def card_to_string(card: int) -> str:
    """Convert card index (0..35) to human-readable form, e.g. '10♥'."""
    if card < 0 or card >= _NUM_CARDS:
        return "None"
    suit_symbols = ["♠", "♣", "♦", "♥"]
    rank_symbols = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    s = suit_of(card)
    r = rank_of(card)
    return f"{rank_symbols[r]}{suit_symbols[s]}"

def card_to_action(card: int) -> int:
    """Actions [0..35] correspond to playing 'card' from hand."""
    return card

def action_to_card(action: int) -> Optional[int]:
    """If the action is in [0..35], interpret as that card index; else None."""
    if 0 <= action < _NUM_CARDS:
        return action
    return None

# Additional actions beyond card indices:
#   TAKE_CARDS: defender picks up all the cards on the table
#   FINISH_ATTACK: attacker stops adding any more attacking cards
#   FINISH_DEFENSE: defender indicates they are done defending if all covered
class ExtraAction(enum.IntEnum):
    TAKE_CARDS = _NUM_CARDS
    FINISH_ATTACK = _NUM_CARDS + 1
    FINISH_DEFENSE = _NUM_CARDS + 2

class RoundPhase(enum.IntEnum):
    """Phases of a Durak round."""
    CHANCE = 0       # Dealing initial cards (and revealing trump)
    ATTACK = 1       # Attacker can place multiple cards before finishing
    DEFENSE = 2      # Defender tries to cover them
    ADDITIONAL = 3   # Attacker can add more cards if all covered so far
    PENDING_TAKE = 4 # Attacker can add more cards after defender decided to take

_GAME_TYPE = pyspiel.GameType(
    short_name="python_durak",
    long_name="Python Durak",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CARDS + 3,  # 36 card-plays + TAKE_CARDS + 2 finishes
    max_chance_outcomes=_NUM_CARDS,       # any card in the deck as a chance outcome
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=300  # a safe upper bound
)

class DurakGame(pyspiel.Game):
    """The Durak game definition for OpenSpiel."""
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        return DurakState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return DurakObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params
        )

class DurakState(pyspiel.State):
    """OpenSpiel state for Durak with multi-card attacking."""
    def __init__(self, game: DurakGame):
        super().__init__(game)
        self._deck = _DECK.copy()
        np.random.shuffle(self._deck)

        # Each player's hand
        self._hands: List[List[int]] = [[] for _ in range(_NUM_PLAYERS)]

        # The table: list of (attacking_card, defending_card_or_None)
        self._table_cards: List[Tuple[int, Optional[int]]] = []

        # Discard pile for covered cards
        self._discard: List[int] = []

        # Trump suit and trump card
        self._trump_suit: Optional[int] = None
        self._trump_card: Optional[int] = None

        # For dealing the initial 6 cards each + revealing trump
        self._cards_dealt = 0
        self._deck_pos = 0

        # Roles
        self._attacker = 0
        self._defender = 1
        self._phase = RoundPhase.CHANCE
        self._round_starter = 0  # Who began the current round as attacker?

        self._game_over = False

    def current_player(self) -> int:
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase == RoundPhase.CHANCE:
            return pyspiel.PlayerId.CHANCE
        if self._phase in [RoundPhase.ATTACK, RoundPhase.ADDITIONAL]:
            return self._attacker
        return self._defender  # RoundPhase.DEFENSE => defender

    def is_terminal(self) -> bool:
        return self._game_over

    def returns(self) -> List[float]:
        if not self._game_over:
            return [0.0, 0.0]
        players_with_cards = [p for p in range(_NUM_PLAYERS) if len(self._hands[p]) > 0]
        if len(players_with_cards) == 1:
            loser = players_with_cards[0]
            result = [0.0, 0.0]
            result[loser] = -1.0
            result[1 - loser] = 1.0
            return result
        if len(players_with_cards) == 2:
            return [0.0, 0.0]
        if self._deck_pos >= len(self._deck):
            result = [0.0, 0.0]
            atk = self._attacker
            result[atk] = 1.0
            result[1 - atk] = -1.0
            return result
        else:
            return [0.0, 0.0]

    def __str__(self) -> str:
        if self._phase == RoundPhase.CHANCE:
            return (f"Chance node: dealing... cards_dealt={self._cards_dealt}, "
                    f"trump_suit={self._trump_suit if self._trump_suit is not None else '??'}")
        lines = []
        lines.append(f"Attacker={self._attacker}, Defender={self._defender}")
        lines.append(f"Phase={RoundPhase(self._phase).name}, Discarded={len(self._discard)}, DeckRemaining={len(self._deck)-self._deck_pos}")
        if self._trump_card is not None:
            lines.append(f"Trump={card_to_string(self._trump_card)} (suit={self._trump_suit})")
        for p in range(_NUM_PLAYERS):
            hand_str = [card_to_string(c) for c in self._hands[p]]
            lines.append(f"Player {p} hand: {hand_str}")
        table_str = []
        for (ac, dc) in self._table_cards:
            if dc is None:
                table_str.append(f"{card_to_string(ac)}->?")
            else:
                table_str.append(f"{card_to_string(ac)}->{card_to_string(dc)}")
        lines.append("Table: " + ", ".join(table_str))
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Chance Node Logic (dealing & revealing trump)
    # --------------------------------------------------------------------------
    def is_chance_node(self) -> bool:
        return (self._phase == RoundPhase.CHANCE)

    def chance_outcomes(self):
        if self._cards_dealt < _CARDS_PER_PLAYER * _NUM_PLAYERS:
            next_card = self._deck[self._deck_pos]
            return [(next_card, 1.0)]
        else:
            if self._trump_card is None:
                trump_card = self._deck[self._deck_pos]
                return [(trump_card, 1.0)]
            return []

    def _apply_chance_action(self, outcome: int):
        if self._cards_dealt < _CARDS_PER_PLAYER * _NUM_PLAYERS:
            player_idx = self._cards_dealt % _NUM_PLAYERS
            self._hands[player_idx].append(outcome)
            self._deck_pos += 1
            self._cards_dealt += 1
        else:
            self._trump_card = self._deck[-1]
            self._trump_suit = suit_of(self._deck[-1])
            self._decide_first_attacker()
            self._phase = RoundPhase.ATTACK
            self._round_starter = self._attacker

    def legal_actions(self):
        """Expose legal actions for the current mover."""
        return self._legal_actions(self.current_player())

    def clone(self):
        """Return a deep copy of the state so that MCTS simulations don't corrupt the original state."""
        import copy
        return copy.deepcopy(self)

    # --------------------------------------------------------------------------
    # Move logic
    # --------------------------------------------------------------------------
    def _legal_actions(self, player: int):
        if self._game_over or self.is_chance_node():
            return []
        actions = []
        hand = self._hands[player]
        uncovered = [(i, ac) for i, (ac, dc) in enumerate(self._table_cards) if dc is None]
        
        if (self._phase == RoundPhase.ATTACK or self._phase == RoundPhase.ADDITIONAL) and (player == self._attacker):
            if len(self._table_cards) == 0:
                for c in hand:
                    actions.append(card_to_action(c))
            elif len(self._table_cards) < _CARDS_PER_PLAYER and len(uncovered) < len(self._hands[self._defender]):
                ranks_on_table = set(rank_of(ac) for (ac, dc) in self._table_cards)
                ranks_on_table.update(rank_of(dc) for (ac, dc) in self._table_cards if dc is not None)
                for c in hand:
                    if rank_of(c) in ranks_on_table:
                        actions.append(card_to_action(c))
            if len(self._table_cards) > 0:
                actions.append(ExtraAction.FINISH_ATTACK)
        
        elif self._phase == RoundPhase.PENDING_TAKE and (player == self._attacker):
            # When defender decided to take cards, attacker can add more cards with matching ranks
            if len(self._table_cards) < len(self._hands[self._defender]):
                ranks_on_table = set(rank_of(ac) for (ac, dc) in self._table_cards)
                ranks_on_table.update(rank_of(dc) for (ac, dc) in self._table_cards if dc is not None)
                for c in hand:
                    if rank_of(c) in ranks_on_table:
                        actions.append(card_to_action(c))
            # Always allow finishing the attack to complete the take
            actions.append(ExtraAction.FINISH_ATTACK)
        
        elif self._phase == RoundPhase.PENDING_TAKE and (player == self._defender):
            # Always allow finishing the defence to complete the take
            actions.append(ExtraAction.FINISH_DEFENSE)
        
        elif self._phase == RoundPhase.DEFENSE and (player == self._defender):
            if len(uncovered) == 0:
                actions.append(ExtraAction.FINISH_DEFENSE)
            else:
                actions.append(ExtraAction.TAKE_CARDS)
                for i, att_card in uncovered:
                    if i == uncovered[0][0]:
                        for c in hand:
                            if self._can_defend_card(c, att_card):
                                actions.append(card_to_action(c))
        
        return sorted(actions)

    def _action_to_string(self, player: int, action: int) -> str:
        if action == ExtraAction.TAKE_CARDS:
            return "TAKE_CARDS"
        elif action == ExtraAction.FINISH_ATTACK:
            return "FINISH_ATTACK"
        elif action == ExtraAction.FINISH_DEFENSE:
            return "FINISH_DEFENSE"
        elif 0 <= action < _NUM_CARDS:
            return f"Play: {card_to_string(action)}"
        else:
            return f"Unknown ({action})"

    def _apply_action(self, action: int):
        if self.is_chance_node():
            self._apply_chance_action(action)
            return
        player = self.current_player()
        
        if action >= _NUM_CARDS:
            if action == ExtraAction.TAKE_CARDS:
                self._defender_takes_cards()
            elif action == ExtraAction.FINISH_ATTACK:
                self._attacker_finishes_attack()
            elif action == ExtraAction.FINISH_DEFENSE:
                self._defender_finishes_defense()
            self._check_game_over()
            return
        
        card = action_to_card(action)
        if card is not None and card in self._hands[player]:
            if (self._phase in [RoundPhase.ATTACK, RoundPhase.ADDITIONAL, RoundPhase.PENDING_TAKE] 
                    and player == self._attacker):
                # Handle attacker playing a card - similar for all attacking phases
                self._hands[player].remove(card)
                self._table_cards.append((card, None))
                if self._phase == RoundPhase.ADDITIONAL:
                    self._phase = RoundPhase.DEFENSE
                # PENDING_TAKE phase remains the same after playing a card
            elif self._phase == RoundPhase.DEFENSE and player == self._defender:
                # Normal defense logic
                uncovered = [(i, ac) for i, (ac, dc) in enumerate(self._table_cards) if dc is None]
                if uncovered:
                    earliest_idx, att_card = uncovered[0]
                    if self._can_defend_card(card, att_card):
                        self._hands[player].remove(card)
                        self._table_cards[earliest_idx] = (att_card, card)
                        if all(dc is not None for (ac, dc) in self._table_cards):
                            self._phase = RoundPhase.ADDITIONAL
        self._check_game_over()

    def _decide_first_attacker(self):
        lowest_trump = None
        who = 0
        for p in range(_NUM_PLAYERS):
            for c in self._hands[p]:
                if suit_of(c) == self._trump_suit:
                    if (lowest_trump is None) or (rank_of(c) < rank_of(lowest_trump)):
                        lowest_trump = c
                        who = p
        self._attacker = who
        self._defender = 1 - who

    def _can_defend_card(self, defense_card: int, attack_card: int) -> bool:
        att_s, att_r = suit_of(attack_card), rank_of(attack_card)
        def_s, def_r = suit_of(defense_card), rank_of(defense_card)
        if att_s == def_s and def_r > att_r:
            return True
        if def_s == self._trump_suit and att_s != self._trump_suit:
            return True
        if att_s == self._trump_suit and def_s == self._trump_suit and def_r > att_r:
            return True
        return False

    def _defender_takes_cards(self):
        # Instead of immediately adding cards to defender's hand,
        # transition to PENDING_TAKE phase where attacker can add more cards
        self._phase = RoundPhase.PENDING_TAKE
        
        # Only proceed with adding cards if there are no uncovered attack cards
        # This allows the normal defense flow to continue
        if not any(dc is None for (ac, dc) in self._table_cards):
            self._complete_take_cards()

    def _complete_take_cards(self):
        # Actually transfer all cards from the table to defender's hand
        for (ac, dc) in self._table_cards:
            self._hands[self._defender].append(ac)
            if dc is not None:
                self._hands[self._defender].append(dc)
        self._table_cards.clear()
        self._phase = RoundPhase.ATTACK
        self._refill_hands()

    def _attacker_finishes_attack(self):
        if len(self._table_cards) == 0:
            return
        
        self._phase = RoundPhase.DEFENSE

    def _defender_finishes_defense(self):
        uncovered = any(dc is None for (ac, dc) in self._table_cards)
        if uncovered:
            self._complete_take_cards()
        else:
            for (ac, dc) in self._table_cards:
                self._discard.append(ac)
                if dc is not None:
                    self._discard.append(dc)
            self._table_cards.clear()
            old_attacker = self._attacker
            self._attacker = self._defender
            self._defender = old_attacker
            self._refill_hands()
            self._phase = RoundPhase.ATTACK

    def _refill_hands(self):
        order = [self._attacker, self._defender]
        while self._deck_pos < len(self._deck):
            for p in order:
                if len(self._hands[p]) < _CARDS_PER_PLAYER and self._deck_pos < len(self._deck):
                    c = self._deck[self._deck_pos]
                    self._deck_pos += 1
                    self._hands[p].append(c)
            if all(len(self._hands[p]) >= _CARDS_PER_PLAYER for p in order):
                break

    def _check_game_over(self):
        if (len(self._hands[0]) == 0 or len(self._hands[1]) == 0) and self._deck_pos >= len(self._deck):
            self._game_over = True
            return
        if (len(self._hands[0]) == 0 and len(self._hands[1]) == 0):
            if self._deck_pos >= len(self._deck):
                self._game_over = True
                return
            else:
                self._refill_hands()

class DurakObserver:
    """Observer for Durak, following the PyObserver interface."""
    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        self._iig_obs_type = iig_obs_type
        pieces = [
            ("player", _NUM_PLAYERS, (_NUM_PLAYERS,)),
            ("trump_suit", 4, (4,)),
            ("phase", 5, (5,)),
            ("deck_size", 1, (1,)),
            ("attacker_ind", 1, (1,)),
            ("defender_ind", 1, (1,)),
            ("trump_card", _NUM_CARDS, (_NUM_CARDS,)),
        ]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("my_cards", _NUM_CARDS, (_NUM_CARDS,)))
        if iig_obs_type.public_info:
            pieces.append(("table_attack", _NUM_CARDS, (_NUM_CARDS,)))
            pieces.append(("table_defense", _NUM_CARDS, (_NUM_CARDS,)))
        total_size = sum(sz for _, sz, _ in pieces)
        self.tensor = np.zeros(total_size, np.float32)
        self.dict = {}
        idx = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[idx : idx + size].reshape(shape)
            idx += size

    def set_from(self, state: "DurakState", player: int):
        self.tensor.fill(0.0)
        if "player" in self.dict:
            self.dict["player"][player] = 1
        if state._trump_suit is not None and "trump_suit" in self.dict:
            self.dict["trump_suit"][state._trump_suit] = 1
        if "trump_card" in self.dict and state._trump_card is not None:
            self.dict["trump_card"][state._trump_card] = 1
        if "phase" in self.dict:
            self.dict["phase"][state._phase] = 1
        if "deck_size" in self.dict:
            ds = (len(state._deck) - state._deck_pos)
            self.dict["deck_size"][0] = ds / float(_NUM_CARDS)
        if "attacker_ind" in self.dict:
            self.dict["attacker_ind"][0] = float(player == state._attacker)
        if "defender_ind" in self.dict:
            self.dict["defender_ind"][0] = float(player == state._defender)
        if "my_cards" in self.dict:
            for c in state._hands[player]:
                self.dict["my_cards"][c] = 1
        if "table_attack" in self.dict and "table_defense" in self.dict:
            for (ac, dc) in state._table_cards:
                self.dict["table_attack"][ac] = 1
                if dc is not None:
                    self.dict["table_defense"][dc] = 1

    def string_from(self, state: "DurakState", player: int) -> str:
        lines = []
        lines.append(f"Player {player} viewpoint")
        lines.append(f"Phase: {RoundPhase(state._phase).name}")
        if state._trump_card is not None:
            lines.append(f"Trump card: {card_to_string(state._trump_card)}")
        lines.append(f"Attacker={state._attacker}, Defender={state._defender}")
        lines.append(f"My hand: {[card_to_string(c) for c in sorted(state._hands[player])]}")
        table_str = []
        for (ac, dc) in state._table_cards:
            if dc is None:
                table_str.append(f"{card_to_string(ac)}->?")
            else:
                table_str.append(f"{card_to_string(ac)}->{card_to_string(dc)}")
        lines.append(f"Table: {table_str}")
        lines.append(f"DeckRemaining: {len(state._deck)-state._deck_pos}")
        return " | ".join(lines)

pyspiel.register_game(_GAME_TYPE, DurakGame)
