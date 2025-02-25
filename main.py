import argparse
import torch
import pyspiel
import numpy as np
from src.durak.durak_game import (
    DurakGame,
    card_to_string,
    ExtraAction,
    DurakObserver,
    RoundPhase,
)
from src.model.network import AlphaZeroNet
from src.model.trainer import Trainer
from src.model.mcts import MCTS
from src.evaluation.rule_agent import RuleAgent
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

def action_to_pretty_str(action):
    """Converts an action index to a human‑readable string."""
    if action < 36:
        return card_to_string(action)
    elif action == ExtraAction.TAKE_CARDS:
        return "Take cards"
    elif action == ExtraAction.FINISH_ATTACK:
        return "Finish attack"
    elif action == ExtraAction.FINISH_DEFENSE:
        return "Finish defense"
    else:
        return f"Unknown action {action}"

def evaluate_state(state, human_player, network, device):
    """
    Evaluates the current state using the network.
    If the current mover is not the human, we flip the value.
    Returns a win probability (0 to 1) for the human.
    """
    observer = DurakObserver(pyspiel.IIGObservationType(
        perfect_recall=False, public_info=True,
        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER), params=None)
    current_player = state.current_player()
    observer.set_from(state, current_player)
    obs = torch.tensor(observer.tensor, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, value = network(obs)
        value = value.item()  # value for the current mover
    if current_player == human_player:
        win_prob = (value + 1) / 2
    else:
        win_prob = ((-value) + 1) / 2
    return win_prob

def play_game(game, network, device, human_player):
    """
    Interactive play mode where the human makes moves.
    """
    state = game.new_initial_state()
    mcts = MCTS(network, num_simulations=50, device=device)

    while not state.is_terminal():
        # Resolve chance nodes.
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            if outcomes:
                outcome, _ = outcomes[0]
                state.apply_action(outcome)
            else:
                break

        win_prob = evaluate_state(state, human_player, network, device)
        print(f"\nWin probability for you: {win_prob*100:.1f}%")

        observer = DurakObserver(pyspiel.IIGObservationType(
            perfect_recall=False, public_info=True,
            private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER), params=None)
        observer.set_from(state, human_player)
        print("\nYour view of the state:")
        print(observer.string_from(state, human_player))
        print("")

        if state.current_player() == human_player:
            legal_actions = state.legal_actions()
            if not legal_actions:
                print("No legal actions available, skipping turn.")
                state.apply_action(ExtraAction.FINISH_ATTACK)
                continue
            print("Your legal moves:")
            for idx, action in enumerate(legal_actions):
                print(f"{idx}: {action_to_pretty_str(action)}")
            valid_choice = False
            while not valid_choice:
                try:
                    choice = int(input("Enter the number of your chosen move: "))
                    if 0 <= choice < len(legal_actions):
                        valid_choice = True
                    else:
                        print("Invalid choice, try again.")
                except ValueError:
                    print("Please enter a valid number.")
            chosen_action = legal_actions[choice]
            print(f"\nYou chose: {action_to_pretty_str(chosen_action)}\n")
            state.apply_action(chosen_action)
        else:
            print("Model is thinking...")
            action_probs = mcts.run(state)
            best_action = max(action_probs, key=action_probs.get)
            print(f"Model chooses: {action_to_pretty_str(best_action)}\n")
            state.apply_action(best_action)

    rewards = state.returns()
    if rewards[human_player] > 0:
        print("You win!")
    elif rewards[human_player] < 0:
        print("You lose!")
    else:
        print("Draw!")

def simulate_game_print(network, device, model_player, mcts_simulations=50):
    """
    Simulates a complete game between the model (using MCTS) and a rule-based agent,
    printing the state and moves at each step for inspection.
    """
    game = DurakGame()
    state = game.new_initial_state()
    mcts = MCTS(network, num_simulations=mcts_simulations, device=device)
    # Rule agent plays for the opponent.
    rule_agent = RuleAgent(player_id=1 - model_player)

    print("=== Simulated Game Start ===")
    move_number = 1
    while not state.is_terminal():
        # Resolve chance nodes.
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            if outcomes:
                outcome, _ = outcomes[0]
                state.apply_action(outcome)
            else:
                break

        current_player = state.current_player()
        print(f"\n--- Move {move_number} ---")
        print(f"Current mover: {'Model' if current_player==model_player else 'Rule Agent'}")
        # Print public view (using the observer from the model's perspective)
        observer = DurakObserver(pyspiel.IIGObservationType(
            perfect_recall=False, public_info=True,
            private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER), params=None)
        # We show the state from the model's perspective so that we don't reveal hidden info.
        observer.set_from(state, model_player)
        print(observer.string_from(state, model_player))
        win_prob = evaluate_state(state, model_player, network, device)
        print(f"Model win probability (from model's turn perspective): {win_prob*100:.1f}%")

        if current_player == model_player:
            action_probs = mcts.run(state)
            chosen_action = max(action_probs, key=action_probs.get)
            print(f"Model plays: {action_to_pretty_str(chosen_action)}")
            state.apply_action(chosen_action)
        else:
            chosen_action = rule_agent.select_action(state)
            if chosen_action is None:
                legal = state.legal_actions()
                chosen_action = legal[0] if legal else None
            print(f"Rule Agent plays: {action_to_pretty_str(chosen_action)}")
            state.apply_action(chosen_action)
        move_number += 1

    rewards = state.returns()
    print("\n=== Game Over ===")
    if rewards[model_player] > 0:
        print("Model wins the game!")
    elif rewards[model_player] < 0:
        print("Rule Agent wins the game!")
    else:
        print("The game is a draw!")
    print("Final state:")
    print(state)

def main():
    parser = argparse.ArgumentParser(description="AlphaZero for Durak with multiple improvements")
    parser.add_argument("--train", action="store_true", help="Run self-play training")
    parser.add_argument("--mixed_training", action="store_true", help="Run mixed training (model vs rule + self-play)")
    parser.add_argument("--warm_start", action="store_true", help="Perform imitation learning from rule agent before RL")
    parser.add_argument("--simulate", action="store_true", help="Simulate a single game for debugging")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--games_per_iter", type=int, default=10, help="Number of games per iteration")
    parser.add_argument("--fraction_vs_rule", type=float, default=0.5, help="Fraction of games vs. rule agent in mixed mode")
    parser.add_argument("--eval_interval", type=int, default=5, help="How often to evaluate vs rule agent")
    parser.add_argument("--eval_games", type=int, default=20, help="How many eval games vs rule agent each time")
    parser.add_argument("--model_player", type=int, default=0, help="Which player the model controls in evaluation/simulation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create game + network
    game = DurakGame()
    network = AlphaZeroNet(
        input_dim=158,
        hidden_dim=256,   # bigger net
        num_actions=40,
        num_layers=4
    ).to(device)

    # Optionally load checkpoint
    if args.checkpoint:
        from src.utils.checkpoint import load_checkpoint
        load_checkpoint(network, args.checkpoint)

    # Create trainer with some recommended defaults
    trainer = Trainer(
        network=network,
        game=game,
        device=device,
        learning_rate=1e-3,
        mcts_simulations=200,   # increased
        c_puct=1.0,
        temperature=1.0,
        use_argmax=False,       # set to True after some training if you like
        take_cards_penalty=0.5, # bigger penalty for 'take cards'
        move_penalty=0.01,
        max_moves=40,           # forcibly end after 40 moves
        forced_terminal_reward=-1.0
    )

    if args.warm_start:
        # Imitation learning from rule agent
        print("Generating rule agent dataset for imitation learning...")
        dataset = trainer.generate_rule_agent_dataset_for_imitation(n_games=500)  # or 1000
        print(f"Dataset size: {len(dataset)}")
        print("Training supervised on rule agent dataset...")
        trainer.train_supervised_on_rule_agent_dataset(dataset, batch_size=64, epochs=3)

    if args.train and not args.mixed_training:
        # pure self-play
        trainer.run_training(
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            batch_size=32,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            model_player=args.model_player
        )
    elif args.mixed_training:
        # half self-play, half vs rule agent
        trainer.run_training_with_mixed_opponents(
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            fraction_vs_rule=args.fraction_vs_rule,
            batch_size=32,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            model_player=args.model_player
        )
    elif args.simulate:
        # quick debug: simulate a single game
        from src.evaluation.rule_agent import RuleAgent
        from src.model.mcts import MCTS
        from src.durak.durak_game import DurakObserver, card_to_string, ExtraAction, RoundPhase

        print("Simulating a single game with the current model vs rule agent...")

        state = game.new_initial_state()
        rule_agent = RuleAgent(player_id=1 - args.model_player)
        mcts = trainer._make_mcts()

        move_number = 1
        while not state.is_terminal():
            while state.is_chance_node():
                outcomes = state.chance_outcomes()
                if outcomes:
                    outcome, _ = outcomes[0]
                    state.apply_action(outcome)
                else:
                    break
                if state.is_terminal():
                    break
            if state.is_terminal():
                break

            current_player = state.current_player()
            print(f"\n--- Move {move_number} ---")
            print(f"Current mover: {'Model' if current_player==args.model_player else 'Rule Agent'}")
            observer = DurakObserver(pyspiel.IIGObservationType(
                perfect_recall=False, public_info=True,
                private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER
            ), params=None)
            observer.set_from(state, args.model_player)
            print(observer.string_from(state, args.model_player))
            win_prob = evaluate_state(state, args.model_player, network, device)
            print(f"Model win probability (from model's turn perspective): {win_prob*100:.1f}%")

            if current_player == args.model_player:
                action_probs = mcts.run(state)
                actions, probs = zip(*action_probs.items())
                chosen_action = np.random.choice(actions, p=probs)
                print(f"Model plays: {card_to_string(chosen_action) if chosen_action<36 else ExtraAction(chosen_action).name}")
                state.apply_action(chosen_action)
            else:
                action = rule_agent.select_action(state)
                if action is None:
                    la = state.legal_actions()
                    if la:
                        action = la[0]
                    else:
                        break
                print(f"Rule Agent plays: {card_to_string(action) if action<36 else ExtraAction(action).name}")
                state.apply_action(action)
            move_number += 1

        print("\n=== Game Over ===")
        rewards = state.returns()
        if rewards[args.model_player] > 0:
            print("Model wins!")
        elif rewards[args.model_player] < 0:
            print("Rule agent wins!")
        else:
            print("Draw!")
        print("Final state:")
        print(state)
    else:
        print("No action specified. Use --train or --mixed_training or --simulate or --warm_start, etc.")

if __name__ == "__main__":
    main()
