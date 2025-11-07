from __future__ import annotations

import numpy as np

from .durak_game import DurakState, discard_seen, table_pairs, talon_count, trump_suit


def _hand_vector(hand: list[int]) -> np.ndarray:
    vec = np.zeros(36, dtype=np.float32)
    for card in hand:
        vec[card] = 1.0
    return vec


def encode_state(state: DurakState, perspective_player: int, truesight: bool = False) -> np.ndarray:
    my_hand = _hand_vector(state.hands[perspective_player])
    if truesight:
        opp_hand = _hand_vector(state.hands[1 - perspective_player])
    else:
        opp_hand = discard_seen(state).astype(np.float32)
    attack_vec = np.zeros(36, dtype=np.float32)
    defense_vec = np.zeros(36, dtype=np.float32)
    for attack_card, defense_card in table_pairs(state):
        attack_vec[attack_card] = 1.0
        if defense_card is not None:
            defense_vec[defense_card] = 1.0
    trump_vec = np.zeros(4, dtype=np.float32)
    trump_vec[trump_suit(state)] = 1.0
    my_hand_count = np.array([len(state.hands[perspective_player]) / 36.0], dtype=np.float32)
    opp_hand_count = np.array([len(state.hands[1 - perspective_player]) / 36.0], dtype=np.float32)
    talon = np.array([talon_count(state) / 36.0], dtype=np.float32)
    role_flags = np.array(
        [
            1.0 if perspective_player == state.attacker else 0.0,
            1.0 if perspective_player == state.defender else 0.0,
        ],
        dtype=np.float32,
    )
    obs = np.concatenate(
        [
            my_hand,
            opp_hand,
            attack_vec,
            defense_vec,
            trump_vec,
            my_hand_count,
            opp_hand_count,
            talon,
            role_flags,
        ],
        dtype=np.float32,
    )
    return obs
