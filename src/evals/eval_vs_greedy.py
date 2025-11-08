from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.agents.policy import masked_argmax
from src.durak.durak_game import (
    ACTION_END_ATTACK,
    ACTION_TAKE_CARDS,
    DurakState,
    apply_action,
    card_rank,
    card_suit,
    current_player,
    initial_state,
    is_terminal,
    legal_actions,
    winner_id,
)
from src.durak.encoding import encode_state


def _legal_indices(state: DurakState) -> np.ndarray:
    return np.flatnonzero(legal_actions(state))


def greedy_policy_action(state: DurakState) -> int:
    mask = legal_actions(state)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise ValueError("No legal actions for greedy agent")
    pid = current_player(state)
    if pid == state.attacker:
        return _greedy_attack(state, legal)
    return _greedy_defend(state, legal)


def _greedy_attack(state: DurakState, legal: np.ndarray) -> int:
    cards = [a for a in legal if a < 36]
    if not state.table:
        non_trump = [c for c in cards if card_suit(c) != state.trump_suit]
        if non_trump:
            return min(non_trump, key=lambda c: card_rank(c))
        if cards:
            return min(cards, key=lambda c: (card_suit(c) == state.trump_suit, card_rank(c)))
    else:
        if ACTION_END_ATTACK in legal and not cards:
            return ACTION_END_ATTACK
        if cards:
            ranks = {card_rank(atk) for atk, _ in state.table}
            ranks.update(card_rank(defn) for _, defn in state.table if defn is not None)
            matching = [c for c in cards if card_rank(c) in ranks]
            if matching:
                non_trump = [c for c in matching if card_suit(c) != state.trump_suit]
                if non_trump:
                    return min(non_trump, key=lambda c: card_rank(c))
                return min(matching, key=lambda c: (card_suit(c) == state.trump_suit, card_rank(c)))
        if ACTION_END_ATTACK in legal:
            return ACTION_END_ATTACK
    return int(legal[0])


def _greedy_defend(state: DurakState, legal: np.ndarray) -> int:
    cards = [a for a in legal if a < 36]
    if cards:
        uncovered = [atk for atk, defense in state.table if defense is None]
        if uncovered:
            target = uncovered[0]
            candidates = [c for c in cards if _can_cover(c, target, state.trump_suit)]
            if candidates:
                non_trump = [c for c in candidates if card_suit(c) != state.trump_suit]
                if non_trump:
                    return min(non_trump, key=lambda c: card_rank(c))
                return min(candidates, key=lambda c: card_rank(c))
    if ACTION_TAKE_CARDS in legal:
        return ACTION_TAKE_CARDS
    return int(legal[0])


def _can_cover(card: int, attack_card: int, trump_suit: int) -> bool:
    suit = card_suit(card)
    atk_suit = card_suit(attack_card)
    if suit == atk_suit:
        return card_rank(card) > card_rank(attack_card)
    if suit == trump_suit and atk_suit != trump_suit:
        return True
    return False


def eval_vs_greedy(qnet: torch.nn.Module, device: torch.device, n_games: int = 200, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    qnet.eval()
    wins = 0
    with torch.no_grad():
        for game_idx in range(n_games):
            state = initial_state(rng)
            learned_player = game_idx % 2
            while not is_terminal(state):
                pid = current_player(state)
                if pid == learned_player:
                    obs = encode_state(state, perspective_player=pid, truesight=False)
                    mask = legal_actions(state)
                    q_values = qnet(torch.from_numpy(obs).to(device).unsqueeze(0)).squeeze(0)
                    action = masked_argmax(q_values, mask)
                else:
                    action = greedy_policy_action(state)
                state, _, _ = apply_action(state, action)
            if winner_id(state) == learned_player:
                wins += 1
    qnet.train()
    return wins / float(n_games)
