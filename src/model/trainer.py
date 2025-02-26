# src/model/trainer.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import pyspiel
from tqdm import tqdm

from src.durak.durak_game import DurakGame, DurakObserver, ExtraAction
from src.model.mcts import MCTS
from src.evaluation.rule_agent import RuleAgent

class Trainer:
    def __init__(
        self, 
        network, 
        game, 
        device="cpu", 
        learning_rate=0.001, 
        mcts_simulations=800, 
        c_puct=1.0, 
        temperature=1.0,
        use_argmax=False,
        checkpoint_dir="checkpoints",
        take_cards_penalty=0.0,  # Optional penalty for taking cards
        move_penalty=0.0,        # Optional small penalty per move to encourage faster play
        max_moves=None,          # Optional max moves per game
        forced_terminal_reward=-1.0,  # Reward if game is forcibly ended
        use_dirichlet=False,     # Whether to use Dirichlet noise in MCTS
        dirichlet_alpha=0.3,     # Alpha parameter for Dirichlet distribution
        dirichlet_epsilon=0.25   # Weight of Dirichlet noise
    ):
        self.network = network
        self.game = game
        self.device = device
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_argmax = use_argmax
        self.checkpoint_dir = checkpoint_dir
        self.take_cards_penalty = take_cards_penalty
        self.move_penalty = move_penalty
        self.max_moves = max_moves
        self.forced_terminal_reward = forced_terminal_reward
        self.game_count = 0
        # Dirichlet noise parameters
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def _make_mcts(self):
        """Create an MCTS instance with current parameters."""
        return MCTS(
            self.network, 
            n_simulations=self.mcts_simulations, 
            c_puct=self.c_puct, 
            temperature=self.temperature,
            use_argmax=self.use_argmax,
            device=self.device,
            use_dirichlet=self.use_dirichlet,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon
        )
    
    def self_play_game(self):
        """
        Play a game against itself and collect training examples.
        Returns a list of (state, policy, value) tuples.
        """
        mcts = self._make_mcts()
        
        state = self.game.new_initial_state()
        training_examples = []
        move_count = 0
        game_history = []  # Track game history for LSTM input
        
        # Handle chance node for initial shuffling and dealing
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = outcomes[0][0]  # Take first action (only one possible)
            state.apply_action(action)
        
        while not state.is_terminal():
            move_count += 1
            
            # Check if we've exceeded max moves
            if self.max_moves and move_count > self.max_moves:
                break
            
            current_player = state.current_player()
            
            # Create history inputs for the network if using history
            history_actions = None
            history_actors = None
            history_lengths = None
            
            if hasattr(self.network, 'use_history') and self.network.use_history:
                # Extract relevant history for current player
                player_actions = []
                player_actors = []
                for actor, action in game_history:
                    # 1 if action by current player, 0 if by opponent
                    is_current = 1 if actor == current_player else 0
                    player_actions.append(action)
                    player_actors.append(is_current)
                
                if player_actions:
                    history_actions = torch.tensor([player_actions], dtype=torch.long, device=self.device)
                    history_actors = torch.tensor([player_actors], dtype=torch.long, device=self.device)
                    history_lengths = torch.tensor([len(player_actions)], dtype=torch.long, device=self.device)
                else:
                    # Empty history
                    history_actions = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                    history_actors = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                    history_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            
            # Get policy from MCTS
            action_probs = mcts.run(state)
            
            # Create the observation for the current player
            observer = DurakObserver(
                pyspiel.IIGObservationType(
                    perfect_recall=False, 
                    public_info=True,
                    private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
                params=None
            )
            observer.set_from(state, current_player)
            
            # Store the state tensor, MCTS policy, and history data
            example = {
                'state': torch.tensor(observer.tensor, dtype=torch.float32),
                'policy': np.zeros(self.game.num_distinct_actions()),
                'value': None,  # Will be updated later
                'current_player': current_player,
                'history_actions': player_actions if hasattr(self.network, 'use_history') and self.network.use_history else None,
                'history_actors': player_actors if hasattr(self.network, 'use_history') and self.network.use_history else None,
                'history_length': len(player_actions) if hasattr(self.network, 'use_history') and self.network.use_history else 0
            }
            
            # Update the policy in the example
            for action, prob in action_probs.items():
                example['policy'][action] = prob
            
            training_examples.append(example)
            
            # Choose an action based on the MCTS policy
            if self.temperature == 0:
                # Greedy choice
                action = max(action_probs.items(), key=lambda x: x[1])[0]
            else:
                # Sample according to the probabilities
                actions, probs = zip(*action_probs.items())
                action = np.random.choice(actions, p=probs)
            
            # Check for take cards penalty
            if action == ExtraAction.TAKE_CARDS and self.take_cards_penalty > 0:
                # Apply penalty for taking cards if configured
                example['take_cards_penalty'] = -self.take_cards_penalty
            
            # Add move to history
            game_history.append((current_player, action))
            
            # Apply the action
            state.apply_action(action)
        
        # Game over, determine rewards
        if state.is_terminal():
            # Use the terminal state rewards for each player
            returns = state.returns()
            for i, example in enumerate(training_examples):
                player = example['current_player']
                example['value'] = returns[player]
        else:
            # Game ended due to move limit - apply forced terminal reward
            for i, example in enumerate(training_examples):
                player = example['current_player']
                example['value'] = self.forced_terminal_reward
        
        # Apply move penalty if configured (to encourage shorter games)
        if self.move_penalty > 0:
            for i, example in enumerate(training_examples):
                # Small penalty for each move
                if 'take_cards_penalty' in example:
                    example['value'] += example['take_cards_penalty']
                example['value'] -= self.move_penalty
        
        self.game_count += 1
        return training_examples
    
    def play_vs_rule_agent(self, model_player=0):
        """
        Play a game against a rule agent and collect training examples.
        model_player: 0 or 1, which player is the model.
        Returns a list of (state, policy, value) tuples for the model player only.
        """
        rule_agent = RuleAgent()
        mcts = self._make_mcts()
        
        state = self.game.new_initial_state()
        training_examples = []
        move_count = 0
        game_history = []  # Track game history for LSTM input
        
        # Handle chance node for initial shuffling and dealing
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            action = outcomes[0][0]  # Take first action (only one possible)
            state.apply_action(action)
        
        while not state.is_terminal():
            move_count += 1
            
            # Check if we've exceeded max moves
            if self.max_moves and move_count > self.max_moves:
                break
            
            current_player = state.current_player()
            
            # Only record training examples for model player
            if current_player == model_player:
                # Create history inputs for the network if using history
                history_actions = None
                history_actors = None
                history_lengths = None
                
                if hasattr(self.network, 'use_history') and self.network.use_history:
                    # Extract relevant history for current player
                    player_actions = []
                    player_actors = []
                    for actor, action in game_history:
                        # 1 if action by current player, 0 if by opponent
                        is_current = 1 if actor == current_player else 0
                        player_actions.append(action)
                        player_actors.append(is_current)
                    
                    if player_actions:
                        history_actions = torch.tensor([player_actions], dtype=torch.long, device=self.device)
                        history_actors = torch.tensor([player_actors], dtype=torch.long, device=self.device)
                        history_lengths = torch.tensor([len(player_actions)], dtype=torch.long, device=self.device)
                    else:
                        # Empty history
                        history_actions = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                        history_actors = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                        history_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
                
                # Get policy from MCTS for model player
                action_probs = mcts.run(state)
                
                # Create the observation for the current player
                observer = DurakObserver(
                    pyspiel.IIGObservationType(
                        perfect_recall=False, 
                        public_info=True,
                        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
                    params=None
                )
                observer.set_from(state, current_player)
                
                # Store training example
                example = {
                    'state': torch.tensor(observer.tensor, dtype=torch.float32),
                    'policy': np.zeros(self.network.num_actions),
                    'value': None,  # Will be updated later
                    'current_player': current_player,
                    'history_actions': player_actions if hasattr(self.network, 'use_history') and self.network.use_history else None,
                    'history_actors': player_actors if hasattr(self.network, 'use_history') and self.network.use_history else None,
                    'history_length': len(player_actions) if hasattr(self.network, 'use_history') and self.network.use_history else 0
                }
                
                # Update the policy in the example
                for action, prob in action_probs.items():
                    example['policy'][action] = prob
                
                training_examples.append(example)
                
                # Choose an action based on the MCTS policy for model player
                if self.temperature == 0:
                    # Greedy choice
                    action = max(action_probs.items(), key=lambda x: x[1])[0]
                else:
                    # Sample according to the probabilities
                    actions, probs = zip(*action_probs.items())
                    action = np.random.choice(actions, p=probs)
                
                # Check for take cards penalty
                if action == ExtraAction.TAKE_CARDS and self.take_cards_penalty > 0:
                    # Apply penalty for taking cards if configured
                    example['take_cards_penalty'] = -self.take_cards_penalty
            else:
                # Rule agent's turn
                action = rule_agent.step(state)
            
            # Add move to history
            game_history.append((current_player, action))
            
            # Apply the action
            state.apply_action(action)
        
        # Game over, determine rewards for model player
        if state.is_terminal():
            # Use the terminal state rewards
            returns = state.returns()
            for example in training_examples:
                example['value'] = returns[model_player]
        else:
            # Game ended due to move limit - apply forced terminal reward
            for example in training_examples:
                example['value'] = self.forced_terminal_reward
        
        # Apply move penalty if configured
        if self.move_penalty > 0:
            for example in training_examples:
                if 'take_cards_penalty' in example:
                    example['value'] += example['take_cards_penalty']
                example['value'] -= self.move_penalty
        
        self.game_count += 1
        return training_examples
    
    def train_step(self, batch):
        """Perform a single training step on a batch of data."""
        # Extract states, policies, and values from the batch
        states_np = np.stack([example["state"] for example in batch])
        target_policies_np = np.stack([example["policy"] for example in batch])
        target_values_np = np.array([example["value"] for example in batch])
        
        # Check policy dimensions and fix if needed (expand to match model's action space)
        if target_policies_np.shape[1] != self.network.num_actions:
            print(f"Warning: Policy dimension mismatch. Got {target_policies_np.shape[1]}, expected {self.network.num_actions}")
            # Expand policies to match the expected size (add zeros for missing actions)
            if target_policies_np.shape[1] < self.network.num_actions:
                padding = np.zeros((target_policies_np.shape[0], 
                                   self.network.num_actions - target_policies_np.shape[1]))
                target_policies_np = np.concatenate([target_policies_np, padding], axis=1)
        
        # Convert to tensors
        states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        target_policies = torch.tensor(target_policies_np, dtype=torch.float32).to(self.device)
        target_values = torch.tensor(target_values_np, dtype=torch.float32).to(self.device)
        
        # Process with history if network supports it
        if hasattr(self.network, 'use_history') and self.network.use_history:
            # Extract history data if available
            has_history = all(["history_actions" in example for example in batch])
            
            if has_history:
                # Find max history length in batch
                max_hist_len = 0
                for example in batch:
                    hist_len = example.get('history_length', 0) or 0  # Handle None
                    if hist_len > max_hist_len:
                        max_hist_len = hist_len
            
                # If no history in dataset or all empty histories
                if max_hist_len == 0:
                    # Forward pass with dummy history
                    dummy_hist_actions = torch.zeros((len(batch), 1), dtype=torch.long).to(self.device)
                    dummy_hist_actors = torch.zeros((len(batch), 1), dtype=torch.long).to(self.device)
                    dummy_hist_lengths = torch.zeros(len(batch), dtype=torch.long).to(self.device)
                    
                    policy_logits, value = self.network(states, dummy_hist_actions, dummy_hist_actors, dummy_hist_lengths)
                else:
                    # Prepare history tensors
                    history_actions = []
                    history_actors = []
                    history_lengths = []
                    
                    for example in batch:
                        h_actions = example.get('history_actions', []) or []
                        h_actors = example.get('history_actors', []) or []
                        h_length = example.get('history_length', 0) or 0
                        
                        # Pad sequences to max_hist_len
                        padded_actions = h_actions + [0] * (max_hist_len - len(h_actions))
                        padded_actors = h_actors + [0] * (max_hist_len - len(h_actors))
                        
                        history_actions.append(padded_actions)
                        history_actors.append(padded_actors)
                        history_lengths.append(h_length)
                    
                    # Convert to tensors
                    history_actions = torch.tensor(history_actions, dtype=torch.long).to(self.device)
                    history_actors = torch.tensor(history_actors, dtype=torch.long).to(self.device)
                    history_lengths = torch.tensor(history_lengths, dtype=torch.long).to(self.device)
                    
                    # Forward pass with history
                    policy_logits, value = self.network(states, history_actions, history_actors, history_lengths)
            else:
                # Forward pass without history
                policy_logits, value = self.network(states)
        else:
            # Forward pass without history
            policy_logits, value = self.network(states)
        
        # Calculate loss - ensure dimensions match
        policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=1)) / target_policies.size(0)
        value_loss = F.mse_loss(value, target_values)  # FIXED: Use value output from network
        total_loss = policy_loss + value_loss
        
        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def generate_rule_agent_dataset_for_imitation(self, n_games=100):
        """Generate a dataset from rule agent gameplay for supervised learning."""
        rule_agent = RuleAgent()
        dataset = []
        
        with tqdm(total=n_games, desc="Generating rule agent dataset") as pbar:
            for _ in range(n_games):
                state = self.game.new_initial_state()
                game_history = []  # Track game history for LSTM input
                
                # Handle chance node for initial shuffling and dealing
                while state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action = outcomes[0][0]  # Take first action (only one possible)
                    state.apply_action(action)
                
                # Play until terminal state is reached
                while not state.is_terminal():
                    current_player = state.current_player()
                    
                    # Get observation tensor for the current state
                    observer = DurakObserver(
                        pyspiel.IIGObservationType(
                            perfect_recall=False, public_info=True,
                            private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER
                        ),
                        params=None
                    )
                    observer.set_from(state, current_player)
                    obs_tensor = observer.tensor
                    
                    # Get rule agent's action
                    action = rule_agent.step(state)
                    
                    # Create target policy (one-hot for the rule agent's action)
                    legal_actions = state.legal_actions()
                    target_policy = np.zeros(self.network.num_actions)
                    if action in legal_actions:
                        target_policy[action] = 1.0
                    else:
                        # If invalid action, create uniform policy over legal actions
                        for a in legal_actions:
                            target_policy[a] = 1.0 / len(legal_actions)
                    
                    # Create history inputs for the network if using history
                    history_actions = []
                    history_actors = []
                    history_length = 0
                    
                    if hasattr(self.network, 'use_history') and self.network.use_history:
                        # Extract relevant history for current player
                        for actor, act in game_history:
                            # 1 if action by current player, 0 if by opponent
                            is_current = 1 if actor == current_player else 0
                            history_actions.append(act)
                            history_actors.append(is_current)
                        
                        history_length = len(history_actions)
                    
                    # Add to dataset
                    example = {
                        'state': obs_tensor,
                        'target_policy': target_policy,
                        'history_actions': history_actions,
                        'history_actors': history_actors,
                        'history_length': history_length
                    }
                    dataset.append(example)
                    
                    # Apply action and update history
                    state.apply_action(action)
                    game_history.append((current_player, action))
                
                pbar.update(1)
        
        return dataset
    
    def train_supervised_on_rule_agent_dataset(self, dataset, batch_size=64, epochs=3):
        """Train the network using supervised learning on a dataset from the rule agent."""
        # Shuffle dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # Check if network uses history
        uses_history = hasattr(self.network, 'use_history') and self.network.use_history
        
        total_loss = 0
        batches = 0
        
        for epoch in range(epochs):
            # Shuffle again for each epoch
            random.shuffle(indices)
            
            for start_idx in range(0, len(dataset), batch_size):
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch = [dataset[i] for i in batch_indices]
                actual_batch_size = len(batch)
                
                # Prepare input tensors - use numpy.stack for better performance
                states_np = np.stack([example['state'] for example in batch])
                states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
                
                target_policies_np = np.stack([example['target_policy'] for example in batch])
                target_policies = torch.tensor(target_policies_np, dtype=torch.float32).to(self.device)
                
                # Process with history if network supports it
                if uses_history:
                    # Find max history length in batch
                    max_hist_len = 0
                    for example in batch:
                        hist_len = example.get('history_length', 0) or 0  # Handle None
                        if hist_len > max_hist_len:
                            max_hist_len = hist_len
                    
                    # If no history in dataset or all empty histories
                    if max_hist_len == 0:
                        # Forward pass with dummy history
                        dummy_hist_actions = torch.zeros((actual_batch_size, 1), dtype=torch.long).to(self.device)
                        dummy_hist_actors = torch.zeros((actual_batch_size, 1), dtype=torch.long).to(self.device)
                        dummy_hist_lengths = torch.zeros(actual_batch_size, dtype=torch.long).to(self.device)
                        
                        policy_logits, _ = self.network(states, dummy_hist_actions, dummy_hist_actors, dummy_hist_lengths)
                    else:
                        # Prepare history tensors
                        history_actions = []
                        history_actors = []
                        history_lengths = []
                        
                        for example in batch:
                            h_actions = example.get('history_actions', []) or []
                            h_actors = example.get('history_actors', []) or []
                            h_length = example.get('history_length', 0) or 0
                            
                            # Pad sequences to max_hist_len
                            padded_actions = h_actions + [0] * (max_hist_len - len(h_actions))
                            padded_actors = h_actors + [0] * (max_hist_len - len(h_actors))
                            
                            history_actions.append(padded_actions)
                            history_actors.append(padded_actors)
                            history_lengths.append(h_length)
                        
                        # Convert to tensors
                        history_actions = torch.tensor(history_actions, dtype=torch.long).to(self.device)
                        history_actors = torch.tensor(history_actors, dtype=torch.long).to(self.device)
                        history_lengths = torch.tensor(history_lengths, dtype=torch.long).to(self.device)
                        
                        # Forward pass with history
                        policy_logits, _ = self.network(states, history_actions, history_actors, history_lengths)
                else:
                    # Forward pass without history
                    policy_logits, _ = self.network(states)
                
                # Calculate cross-entropy loss
                policy_loss = -torch.sum(target_policies * F.log_softmax(policy_logits, dim=1)) / target_policies.size(0)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
                
                total_loss += policy_loss.item()
                batches += 1
            
            avg_loss = total_loss / batches if batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Supervised Loss: {avg_loss:.4f}")
    
    def save_checkpoint(self, iteration=0):
        """Save a checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{iteration}.ckpt")
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'game_count': self.game_count
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.game_count = checkpoint.get('game_count', 0)
        iteration = checkpoint.get('iteration', 0)
        print(f"Loaded checkpoint from {checkpoint_path} (iteration {iteration})")
        return iteration
    
    def run_training(self, num_iterations=1000, batch_size=32, checkpoint_interval=100):
        """Run the full training loop."""
        for iteration in range(num_iterations):
            # Self-play to generate data
            data = []
            for _ in range(32):  # Number of games per iteration
                game_data = self.self_play_game()
                data.extend(game_data)
            
            # Shuffle the data
            random.shuffle(data)
            
            # Train in batches
            num_batches = (len(data) + batch_size - 1) // batch_size
            total_loss = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                batch = data[start_idx:end_idx]
                
                loss, _, _ = self.train_step(batch)
                total_loss += loss
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {avg_loss:.4f}, Games: {self.game_count}")
            
            # Save checkpoint
            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(iteration + 1)
    
    def run_training_with_mixed_opponents(self, num_iterations=1000, batch_size=32, checkpoint_interval=100,
                                        games_per_iteration=32, fraction_vs_rule=0.5):
        """Run training with a mix of self-play and games against the rule agent."""
        for iteration in range(num_iterations):
            # Generate data from mix of self-play and games vs rule agent
            data = []
            num_vs_rule = int(games_per_iteration * fraction_vs_rule)
            num_self = games_per_iteration - num_vs_rule
            
            # Self-play games
            for _ in range(num_self):
                game_data = self.self_play_game()
                data.extend(game_data)
            
            # Games vs rule agent
            for _ in range(num_vs_rule):
                game_data = self.play_vs_rule_agent()
                data.extend(game_data)
            
            # Shuffle the data
            random.shuffle(data)
            
            # Train in batches
            num_batches = (len(data) + batch_size - 1) // batch_size
            total_loss = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data))
                batch = data[start_idx:end_idx]
                
                loss, _, _ = self.train_step(batch)
                total_loss += loss
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {avg_loss:.4f}, Games: {self.game_count}")
            
            # Save checkpoint
            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint(iteration + 1)