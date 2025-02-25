# src/utils/play_utils.py
import numpy as np
import torch
import pyspiel
from src.durak.durak_game import DurakGame, DurakObserver, card_to_string
from src.model.mcts import MCTS

def get_model_move(network, state, device="cpu", mcts_simulations=800, temperature=0.0, use_argmax=True):
    """Get the model's move for a given state."""
    mcts = MCTS(
        network=network,
        n_simulations=mcts_simulations,
        c_puct=1.0,
        temperature=temperature,
        use_argmax=use_argmax,
        device=device
    )
    
    policy = mcts.run(state)
    action = max(policy.items(), key=lambda x: x[1])[0]
    
    # Also get the evaluation
    observer = DurakObserver(
        pyspiel.IIGObservationType(
            perfect_recall=False, public_info=True,
            private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
        params=None
    )
    observer.set_from(state, state.current_player())
    obs = torch.tensor(observer.tensor, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        _, value = network(obs)
        value = value.item()
    
    # Convert value [-1, 1] to win probability [0, 1]
    win_prob = (value + 1) / 2
    
    return action, policy, win_prob

def print_state_info(state, player_viewpoint):
    """Print readable information about the current state."""
    print(f"\nPlayer {player_viewpoint} viewpoint:")
    
    if state._trump_card is not None:
        print(f"Trump suit: {['♠', '♣', '♦', '♥'][state._trump_suit]} (card: {card_to_string(state._trump_card)})")
    
    print(f"Hand: {[card_to_string(c) for c in sorted(state._hands[player_viewpoint])]}")
    
    print(f"Opponent has {len(state._hands[1-player_viewpoint])} cards")
    print(f"Deck has {len(state._deck) - state._deck_pos} cards remaining")
    
    if len(state._table_cards) > 0:
        print("Table:")
        for i, (ac, dc) in enumerate(state._table_cards):
            if dc is None:
                print(f"  {i+1}. {card_to_string(ac)} -> ?")
            else:
                print(f"  {i+1}. {card_to_string(ac)} -> {card_to_string(dc)}")
    else:
        print("Table: empty")
    
    phase_names = ["CHANCE", "ATTACK", "DEFENSE", "ADDITIONAL", "PENDING_TAKE"]
    print(f"Phase: {phase_names[state._phase]}")
    print(f"Current player: {state.current_player()}")
    
    legal_actions = state.legal_actions()
    print(f"Legal actions: {[action_to_readable(a) for a in legal_actions]}")

def action_to_readable(action):
    """Convert an action to a readable string."""
    if action < 36:
        return card_to_string(action)
    elif action == 36:
        return "TAKE_CARDS"
    elif action == 37:
        return "FINISH_ATTACK"
    elif action == 38:
        return "FINISH_DEFENSE"
    return str(action)

def create_custom_state(trump_card, hands, table_cards, phase, attacker, deck_size=0):
    """Create a custom Durak state for evaluation."""
    game = DurakGame()
    state = game.new_initial_state()
    
    # Set the trump suit and card
    state._trump_card = trump_card
    state._trump_suit = trump_card // 9
    
    # Set the hands
    state._hands = hands
    
    # Set the table cards
    state._table_cards = table_cards
    
    # Set the phase
    state._phase = phase
    
    # Set attacker and defender
    state._attacker = attacker
    state._defender = 1 - attacker
    
    # Set deck size
    remaining_cards = list(range(36))
    for hand in hands:
        for card in hand:
            if card in remaining_cards:
                remaining_cards.remove(card)
    
    for ac, dc in table_cards:
        if ac in remaining_cards:
            remaining_cards.remove(ac)
        if dc is not None and dc in remaining_cards:
            remaining_cards.remove(dc)
    
    if trump_card in remaining_cards:
        remaining_cards.remove(trump_card)
    
    # Create a deck with cards not in hands or on table
    state._deck = remaining_cards
    state._deck_pos = max(0, len(remaining_cards) - deck_size)
    
    return state

def play_from_position(network, trump_card, hands, table_cards, phase, attacker, deck_size=0, device="cpu"):
    """Play from a custom position with the trained model."""
    state = create_custom_state(trump_card, hands, table_cards, phase, attacker, deck_size)
    
    print_state_info(state, attacker)
    
    if not state.is_terminal() and not state.is_chance_node():
        action, policy, win_prob = get_model_move(
            network, state, device=device, mcts_simulations=800, temperature=0.0, use_argmax=True
        )
        
        print(f"\nModel's evaluation: {win_prob:.2%} chance of winning")
        print(f"Model's chosen action: {action_to_readable(action)}")
        
        # Print top 3 actions by probability
        print("\nTop actions by probability:")
        sorted_actions = sorted(policy.items(), key=lambda x: x[1], reverse=True)
        for i, (act, prob) in enumerate(sorted_actions[:3]):
            print(f"  {i+1}. {action_to_readable(act)}: {prob:.2%}")