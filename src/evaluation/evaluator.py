# src/evaluation/evaluator.py
import torch
import pyspiel
import numpy as np
from tqdm import tqdm
from src.durak.durak_game import DurakGame, DurakObserver, card_to_string, ExtraAction
from src.model.mcts import MCTS
from src.evaluation.rule_agent import RuleAgent

def evaluate_model_vs_rule_agent(network, device, num_games=100, model_player=0, mcts_simulations=800, temperature=0.0, use_argmax=True, verbose=False):
    """Evaluate the model against a rule agent with detailed statistics."""
    rule_agent = RuleAgent()
    game = DurakGame()
    
    mcts = MCTS(
        network=network,
        n_simulations=mcts_simulations,
        c_puct=1.0,
        temperature=temperature,
        use_argmax=use_argmax,
        device=device
    )
    
    model_wins = 0
    rule_wins = 0
    draws = 0
    
    # Additional statistics
    total_moves = 0
    model_card_plays = 0  # card play moves
    model_take_cards = 0  # times the model took cards
    rule_take_cards = 0   # times rule agent took cards
    
    # Game length statistics
    game_lengths = []
    
    for game_idx in tqdm(range(num_games), desc="Evaluating", disable=not verbose):
        state = game.new_initial_state()
        move_count = 0
        
        # Handle chance node for initial shuffling and dealing
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = outcomes[0][0]  # Take first action (only one possible)
            state.apply_action(action)
        
        while not state.is_terminal():
            move_count += 1
            
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = outcomes[0][0]  # Take first action (only one possible)
                state.apply_action(action)
                continue
                
            current_player = state.current_player()
            
            if current_player == model_player:
                # Model's turn
                policy = mcts.run(state)
                
                if not policy:  # If empty (shouldn't happen, but just in case)
                    legal_actions = state.legal_actions()
                    if legal_actions:
                        action = legal_actions[0]
                    else:
                        break
                else:
                    action = max(policy.items(), key=lambda x: x[1])[0]
                
                # Track model actions
                if action < 36:  # Card play
                    model_card_plays += 1
                elif action == ExtraAction.TAKE_CARDS:
                    model_take_cards += 1
            else:
                # Rule agent's turn
                action = rule_agent.step(state)
                
                # Track rule agent actions
                if action == ExtraAction.TAKE_CARDS:
                    rule_take_cards += 1
            
            # Apply the action
            state.apply_action(action)
            total_moves += 1
        
        # Game over, determine winner
        if state.is_terminal():
            returns = state.returns()
            if returns[model_player] > 0:
                model_wins += 1
            elif returns[model_player] < 0:
                rule_wins += 1
            else:
                draws += 1
        
        # Record game length
        game_lengths.append(move_count)
    
    model_win_rate = model_wins / num_games
    rule_win_rate = rule_wins / num_games
    draw_rate = draws / num_games
    
    # Calculate average game length
    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    
    if verbose:
        print(f"Model wins: {model_wins} ({model_win_rate:.2%})")
        print(f"Rule agent wins: {rule_wins} ({rule_win_rate:.2%})")
        print(f"Draws: {draws} ({draw_rate:.2%})")
        print(f"Average game length: {avg_game_length:.1f} moves")
        print(f"Model took cards {model_take_cards} times")
        print(f"Rule agent took cards {rule_take_cards} times")
    
    # Return detailed statistics as a dictionary
    stats = {
        'model_win_rate': model_win_rate,
        'rule_win_rate': rule_win_rate,
        'draw_rate': draw_rate,
        'avg_game_length': avg_game_length,
        'model_take_cards': model_take_cards,
        'rule_take_cards': rule_take_cards,
        'model_wins': model_wins,
        'rule_wins': rule_wins,
        'draws': draws
    }
    
    return model_win_rate  # For backward compatibility