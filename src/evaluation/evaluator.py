# src/evaluation/evaluator.py
import torch
import pyspiel
from src.durak.durak_game import DurakGame, DurakObserver
from src.model.mcts import MCTS
from src.evaluation.rule_agent import RuleAgent

def evaluate_model_vs_rule_agent(network, device, num_games=100, model_player=0, mcts_simulations=800, temperature=0.0, use_argmax=True):
    """Evaluate the model against a rule agent."""
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
    
    for game_idx in range(num_games):
        state = game.new_initial_state()
        
        # Handle chance node for initial shuffling and dealing
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = outcomes[0][0]  # Take first action (only one possible)
            state.apply_action(action)
        
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = outcomes[0][0]  # Take first action (only one possible)
                state.apply_action(action)
                continue
                
            current_player = state.current_player()
            
            if current_player == model_player:
                # Model's turn
                policy = mcts.run(state)
                action = max(policy.items(), key=lambda x: x[1])[0]
            else:
                # Rule agent's turn
                action = rule_agent.step(state)
            
            # Apply the action
            state.apply_action(action)
        
        # Game over, determine winner
        if state.is_terminal():
            returns = state.returns()
            if returns[model_player] > 0:
                model_wins += 1
            elif returns[model_player] < 0:
                rule_wins += 1
            else:
                draws += 1
    
    model_win_rate = model_wins / num_games
    rule_win_rate = rule_wins / num_games
    draw_rate = draws / num_games
    
    print(f"Model wins: {model_wins} ({model_win_rate:.2%})")
    print(f"Rule agent wins: {rule_wins} ({rule_win_rate:.2%})")
    print(f"Draws: {draws} ({draw_rate:.2%})")
    
    return model_win_rate