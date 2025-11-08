from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

SUITS = ["clubs", "diamonds", "hearts", "spades"]
RANKS = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]
NUM_PLAYERS = 2
NUM_CARDS = len(SUITS) * len(RANKS)
MAX_ATTACK_CARDS = 6
ACTION_END_ATTACK = 36
ACTION_TAKE_CARDS = 37


@dataclass
class DurakState:
    hands: List[List[int]]
    talon: List[int]
    discard: List[int]
    table: List[Tuple[int, Optional[int]]]
    attacker: int
    defender: int
    phase: str  # "attack" or "defense"
    trump_suit: int
    trump_card: int
    seen_cards: set[int] = field(default_factory=set)
    round_attack_limit: int = MAX_ATTACK_CARDS
    terminal: bool = False
    winner: Optional[int] = None
    last_round_winner: Optional[int] = None
    defender_taking: bool = False
    post_take_additions_remaining: int = 0

    def copy(self) -> "DurakState":
        return DurakState(
            hands=[hand.copy() for hand in self.hands],
            talon=self.talon.copy(),
            discard=self.discard.copy(),
            table=[(atk, defn) for atk, defn in self.table],
            attacker=self.attacker,
            defender=self.defender,
            phase=self.phase,
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            seen_cards=self.seen_cards.copy(),
            round_attack_limit=self.round_attack_limit,
            terminal=self.terminal,
            winner=self.winner,
            last_round_winner=self.last_round_winner,
            defender_taking=self.defender_taking,
            post_take_additions_remaining=self.post_take_additions_remaining,
        )


def card_to_id(card: Tuple[int, int]) -> int:
    suit, rank = card
    return suit * len(RANKS) + rank


def id_to_card(card_id: int) -> Tuple[int, int]:
    suit = card_id // len(RANKS)
    rank = card_id % len(RANKS)
    return suit, rank


def card_suit(card_id: int) -> int:
    return card_id // len(RANKS)


def card_rank(card_id: int) -> int:
    return card_id % len(RANKS)


def initial_state(rng: np.random.Generator) -> DurakState:
    deck = rng.permutation(NUM_CARDS).tolist()
    hands: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
    for _ in range(6):
        for pid in range(NUM_PLAYERS):
            hands[pid].append(deck.pop())
    trump_card = deck.pop()
    trump_suit = card_suit(trump_card)
    talon = deck
    talon.append(trump_card)
    for hand in hands:
        hand.sort()
    attacker = _choose_initial_attacker(hands, trump_suit)
    defender = 1 - attacker
    seen_cards = {trump_card}
    state = DurakState(
        hands=hands,
        talon=talon,
        discard=[],
        table=[],
        attacker=attacker,
        defender=defender,
        phase="attack",
        trump_suit=trump_suit,
        trump_card=trump_card,
        seen_cards=seen_cards,
        round_attack_limit=_compute_attack_limit(hands[defender]),
    )
    state.last_round_winner = None
    return state


def _choose_initial_attacker(hands: Sequence[Sequence[int]], trump_suit: int) -> int:
    lowest_trumps: List[Tuple[int, int]] = []
    for pid, hand in enumerate(hands):
        trumps = [card for card in hand if card_suit(card) == trump_suit]
        if trumps:
            lowest = min(trumps, key=card_rank)
            lowest_trumps.append((card_rank(lowest), pid))
    if lowest_trumps:
        _, pid = min(lowest_trumps)
        return pid
    return 0


def _compute_attack_limit(defender_hand: Sequence[int]) -> int:
    return min(MAX_ATTACK_CARDS, max(1, len(defender_hand)))


def legal_actions(state: DurakState) -> np.ndarray:
    mask = np.zeros(ACTION_TAKE_CARDS + 1, dtype=np.bool_)
    if state.terminal:
        return mask
    if state.phase == "attack":
        attacker_hand = state.hands[state.attacker]
        if state.defender_taking:
            playable: List[int] = []
            if (
                state.post_take_additions_remaining > 0
                and len(state.table) < MAX_ATTACK_CARDS
            ):
                ranks_on_table = _ranks_on_table(state.table)
                playable = [card for card in attacker_hand if card_rank(card) in ranks_on_table]
            for card in playable:
                mask[card] = True
            mask[ACTION_END_ATTACK] = True
        elif state.table:
            if _all_cards_covered(state.table):
                if len(state.table) < state.round_attack_limit and attacker_hand:
                    ranks_on_table = _ranks_on_table(state.table)
                    playable = [card for card in attacker_hand if card_rank(card) in ranks_on_table]
                else:
                    playable = []
                for card in playable:
                    mask[card] = True
                if state.table and _all_cards_covered(state.table):
                    mask[ACTION_END_ATTACK] = True
            else:
                # Should not happen: attacker never gets turn while cards uncovered.
                pass
        else:
            for card in attacker_hand:
                mask[card] = True
        mask[ACTION_TAKE_CARDS] = False
        if not state.table and not attacker_hand:
            mask[:] = False
    else:  # defense
        defender_hand = state.hands[state.defender]
        uncovered = _uncovered_attack_cards(state.table)
        if uncovered:
            targets = [atk for _, atk in uncovered]
            for card in defender_hand:
                if any(_can_cover(card, atk, state.trump_suit) for atk in targets):
                    mask[card] = True
            mask[ACTION_TAKE_CARDS] = True
        if state.table and not uncovered:
            # Defender finished covering, awaiting attacker decision.
            mask[:] = False
        mask[ACTION_END_ATTACK] = False
    return mask


def apply_action(state: DurakState, action: int) -> Tuple[DurakState, bool, Optional[int]]:
    if state.terminal:
        return state, True, state.winner
    if state.phase == "attack":
        if action == ACTION_END_ATTACK:
            if state.defender_taking:
                _finalize_take(state)
            elif state.table and _all_cards_covered(state.table):
                _finish_round_with_defense(state)
            else:
                raise ValueError("Cannot end attack before defender covers all cards.")
        else:
            _play_attack_card(state, action)
    else:
        if action == ACTION_TAKE_CARDS:
            _defender_takes(state)
        else:
            _defend_card(state, action)
    _check_terminal(state)
    return state, state.terminal, state.winner


def _play_attack_card(state: DurakState, card: int) -> None:
    if card not in state.hands[state.attacker]:
        raise ValueError("Attacker does not hold this card.")
    if state.table:
        if state.defender_taking:
            if len(state.table) >= MAX_ATTACK_CARDS:
                raise ValueError("Attack limit reached for this round.")
            if state.post_take_additions_remaining <= 0:
                raise ValueError("No additional cards allowed after defender takes.")
            ranks = _ranks_on_table(state.table)
            if card_rank(card) not in ranks:
                raise ValueError("Attack card must match rank already on table.")
        else:
            if not _all_cards_covered(state.table):
                raise ValueError("Cannot add new attack card until defender covers current cards.")
            if len(state.table) >= state.round_attack_limit:
                raise ValueError("Attack limit reached for this round.")
            ranks = _ranks_on_table(state.table)
            if card_rank(card) not in ranks:
                raise ValueError("Attack card must match rank already on table.")
    state.hands[state.attacker].remove(card)
    state.table.append((card, None))
    state.seen_cards.add(card)
    if state.defender_taking:
        state.post_take_additions_remaining = max(0, state.post_take_additions_remaining - 1)
        if (
            state.post_take_additions_remaining == 0
            or len(state.table) >= MAX_ATTACK_CARDS
            or not _attacker_has_matching_rank(state)
        ):
            _finalize_take(state)
        return
    state.phase = "defense"


def _defend_card(state: DurakState, card: int) -> None:
    if card not in state.hands[state.defender]:
        raise ValueError("Defender does not hold this card.")
    uncovered = _uncovered_attack_cards(state.table)
    if not uncovered:
        raise ValueError("No cards to defend against.")
    cover_index = None
    for idx, attack_card in uncovered:
        if _can_cover(card, attack_card, state.trump_suit):
            cover_index = idx
            break
    if cover_index is None:
        raise ValueError("Card cannot cover any attack card.")
    state.hands[state.defender].remove(card)
    attack_card, _ = state.table[cover_index]
    state.table[cover_index] = (attack_card, card)
    state.seen_cards.add(card)
    if _all_cards_covered(state.table):
        if (
            len(state.table) < state.round_attack_limit
            and state.hands[state.attacker]
            and state.hands[state.defender]
            and _attacker_has_matching_rank(state)
        ):
            state.phase = "attack"
        else:
            _finish_round_with_defense(state)
    else:
        state.phase = "defense"


def _defender_takes(state: DurakState) -> None:
    if state.defender_taking:
        return
    state.defender_taking = True
    state.post_take_additions_remaining = min(
        MAX_ATTACK_CARDS, len(state.hands[state.defender])
    )
    state.phase = "attack"
    if (
        state.post_take_additions_remaining == 0
        or len(state.table) >= MAX_ATTACK_CARDS
        or not _attacker_has_matching_rank(state)
    ):
        _finalize_take(state)
    state.last_round_winner = None


def _finish_round_with_defense(state: DurakState) -> None:
    for attack_card, defense_card in state.table:
        state.discard.append(attack_card)
        state.seen_cards.add(attack_card)
        if defense_card is not None:
            state.discard.append(defense_card)
            state.seen_cards.add(defense_card)
    state.table.clear()
    old_attacker = state.attacker
    old_defender = state.defender
    _refill_hands(state, old_attacker, old_defender)
    state.attacker = old_defender
    state.defender = old_attacker
    state.phase = "attack"
    state.round_attack_limit = _compute_attack_limit(state.hands[state.defender])
    state.hands[state.attacker].sort()
    state.hands[state.defender].sort()
    state.last_round_winner = state.attacker
    state.defender_taking = False
    state.post_take_additions_remaining = 0


def _finalize_take(state: DurakState) -> None:
    for attack_card, defense_card in state.table:
        state.hands[state.defender].append(attack_card)
        if defense_card is not None:
            state.hands[state.defender].append(defense_card)
    state.hands[state.defender].sort()
    state.table.clear()
    _refill_hands(state, state.attacker, state.defender)
    state.phase = "attack"
    state.round_attack_limit = _compute_attack_limit(state.hands[state.defender])
    state.defender_taking = False
    state.post_take_additions_remaining = 0
    state.last_round_winner = None


def _refill_hands(state: DurakState, first: int, second: int) -> None:
    for pid in (first, second):
        while len(state.hands[pid]) < 6 and state.talon:
            card = state.talon.pop(0)
            state.hands[pid].append(card)
        state.hands[pid].sort()


def _attacker_has_matching_rank(state: DurakState) -> bool:
    ranks = _ranks_on_table(state.table)
    for card in state.hands[state.attacker]:
        if card_rank(card) in ranks:
            return True
    return False


def _uncovered_attack_cards(table: Sequence[Tuple[int, Optional[int]]]) -> List[Tuple[int, int]]:
    return [(idx, atk) for idx, (atk, defense) in enumerate(table) if defense is None]


def _all_cards_covered(table: Sequence[Tuple[int, Optional[int]]]) -> bool:
    return bool(table) and all(defense is not None for _, defense in table)


def _can_cover(defense_card: int, attack_card: int, trump_suit: int) -> bool:
    defense_suit = card_suit(defense_card)
    attack_suit = card_suit(attack_card)
    if defense_suit == attack_suit:
        return card_rank(defense_card) > card_rank(attack_card)
    if defense_suit == trump_suit and attack_suit != trump_suit:
        return True
    return False


def _ranks_on_table(table: Sequence[Tuple[int, Optional[int]]]) -> set[int]:
    ranks = {card_rank(atk) for atk, _ in table}
    ranks.update(card_rank(defense) for _, defense in table if defense is not None)
    return ranks


def _check_terminal(state: DurakState) -> None:
    if state.terminal:
        return
    if state.table:
        return
    if state.talon:
        return
    hand_sizes = [len(hand) for hand in state.hands]
    zero_players = [pid for pid, size in enumerate(hand_sizes) if size == 0]
    if len(zero_players) == 1:
        state.terminal = True
        state.winner = zero_players[0]
    elif len(zero_players) == 2:
        winner = state.last_round_winner if state.last_round_winner is not None else state.attacker
        state.terminal = True
        state.winner = winner


def current_player(state: DurakState) -> int:
    return state.attacker if state.phase == "attack" else state.defender


def is_terminal(state: DurakState) -> bool:
    return state.terminal


def winner_id(state: DurakState) -> Optional[int]:
    return state.winner


def is_attacker(state: DurakState, player: Optional[int] = None) -> bool:
    pid = current_player(state) if player is None else player
    return pid == state.attacker


def is_defender(state: DurakState, player: Optional[int] = None) -> bool:
    pid = current_player(state) if player is None else player
    return pid == state.defender


def hand_counts(state: DurakState) -> Tuple[int, int]:
    return len(state.hands[0]), len(state.hands[1])


def talon_count(state: DurakState) -> int:
    return len(state.talon)


def trump_suit(state: DurakState) -> int:
    return state.trump_suit


def table_pairs(state: DurakState) -> List[Tuple[int, Optional[int]]]:
    return list(state.table)


def discard_seen(state: DurakState) -> np.ndarray:
    seen = np.zeros(NUM_CARDS, dtype=np.bool_)
    for card in state.discard:
        seen[card] = True
    for attack_card, defense_card in state.table:
        seen[attack_card] = True
        if defense_card is not None:
            seen[defense_card] = True
    for card in state.seen_cards:
        seen[card] = True
    return seen
