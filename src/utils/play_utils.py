# src/utils/play_utils.py
import numpy as np
import torch
import pyspiel
from src.durak.durak_game import DurakGame, DurakObserver, card_to_string, ExtraAction
from src.model.mcts import MCTS

def get_model_move(network, state, device="cpu", mcts_simulations=800, temperature=0.0, use_argmax=True, use_dirichlet=False):
    """Get the model's move for a given state with enhanced options."""
    mcts = MCTS(
        network=network,
        n_simulations=mcts_simulations,
        c_puct=1.0,
        temperature=temperature,
        use_argmax=use_argmax,
        device=device,
        use_dirichlet=use_dirichlet  # Enable for more exploration during analysis
    )
    
    policy = mcts.run(state)
    
    if not policy:
        legal_actions = state.legal_actions()
        if legal_actions:
            action = legal_actions[0]
            return action, {action: 1.0}, 0.5  # Default 50% win probability
        return None, {}, 0.5
    
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

def print_state_info(state, player_viewpoint, show_legal_actions=True, show_value=False, network=None, device="cpu"):
    """Print readable information about the current state with optional network evaluation."""
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
    
    if show_legal_actions:
        legal_actions = state.legal_actions()
        print(f"Legal actions: {[action_to_readable(a) for a in legal_actions]}")
    
    # Show network evaluation if requested
    if show_value and network is not None:
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
        
        current_player = state.current_player()
        if current_player != player_viewpoint:
            win_prob = 1 - win_prob  # Invert from other player's perspective
            
        print(f"Estimated win probability: {win_prob:.2%}")

def action_to_readable(action):
    """Convert an action to a readable string."""
    if action < 36:
        return card_to_string(action)
    elif action == ExtraAction.TAKE_CARDS:
        return "TAKE_CARDS"
    elif action == ExtraAction.FINISH_ATTACK:
        return "FINISH_ATTACK"
    elif action == ExtraAction.FINISH_DEFENSE:
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

def play_interactive_game(network, starting_state=None, device="cpu", human_player=0):
    """Play an interactive game where human plays against the model."""
    if starting_state is None:
        game = DurakGame()
        state = game.new_initial_state()
        
        # Handle chance node for initial shuffling and dealing
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = outcomes[0][0]
            state.apply_action(action)
    else:
        state = starting_state.clone()
    
    print("=== Interactive Durak Game vs Model ===")
    print("You are player", human_player)
    
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = outcomes[0][0]
            state.apply_action(action)
            continue
        
        current_player = state.current_player()
        print_state_info(state, human_player, network=network, device=device, show_value=True)
        
        if current_player == human_player:
            # Human's turn
            legal_actions = state.legal_actions()
            if not legal_actions:
                print("No legal actions available.")
                break
            
            print("\nYour turn - choose an action:")
            for i, action in enumerate(legal_actions):
                print(f"{i+1}: {action_to_readable(action)}")
            
            # Get human input
            choice = -1
            while choice < 0 or choice >= len(legal_actions):
                try:
                    choice = int(input("Enter action number: ")) - 1
                    if choice < 0 or choice >= len(legal_actions):
                        print(f"Please enter a number between 1 and {len(legal_actions)}")
                except ValueError:
                    print("Please enter a valid number")
            
            action = legal_actions[choice]
            print(f"You chose: {action_to_readable(action)}")
        else:
            # Model's turn
            print("\nModel is thinking...")
            action, policy, win_prob = get_model_move(network, state, device=device)
            print(f"Model's evaluation: {win_prob:.2%} chance of winning")
            print(f"Model chose: {action_to_readable(action)}")
            
            # Optionally show top alternatives
            sorted_actions = sorted(policy.items(), key=lambda x: x[1], reverse=True)[:3]
            if len(sorted_actions) > 1:
                print("Top alternatives considered:")
                for i, (act, prob) in enumerate(sorted_actions[1:], 1):
                    print(f"  {i}. {action_to_readable(act)}: {prob:.2%}")
        
        # Apply the chosen action
        state.apply_action(action)
        print("\n" + "="*40 + "\n")
    
    # Game over
    print("Game Over!")
    if state.is_terminal():
        returns = state.returns()
        if returns[human_player] > 0:
            print("You win!")
        elif returns[human_player] < 0:
            print("Model wins!")
        else:
            print("It's a draw!")
    else:
        print("Game ended in a non-terminal state.")