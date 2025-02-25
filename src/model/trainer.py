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
        forced_terminal_reward=-1.0  # Reward if game is forcibly ended
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
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def self_play_game(self):
        """
        Play a game against itself and collect training examples.
        Returns a list of (state, policy, value) tuples.
        """
        mcts = MCTS(
            self.network, 
            n_simulations=self.mcts_simulations, 
            c_puct=self.c_puct, 
            temperature=self.temperature,
            use_argmax=self.use_argmax,
            device=self.device
        )
        
        state = self.game.new_initial_state()
        training_examples = []
        move_count = 0
        
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
            
            # Get policy from MCTS
            action_probs = mcts.run(state)
            
            # Handle empty policy case
            if not action_probs:
                legal_actions = state.legal_actions()
                if legal_actions:
                    # If we have legal actions but empty policy, create uniform policy
                    action_probs = {action: 1.0/len(legal_actions) for action in legal_actions}
                else:
                    # No legal actions, must be a terminal state or chance node
                    continue
                
            # Create the observer
            observer = DurakObserver(
                pyspiel.IIGObservationType(
                    perfect_recall=False, public_info=True,
                    private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
                params=None
            )
            
            # Set the observation from the current player's perspective
            observer.set_from(state, current_player)
            
            # Choose an action based on the policy
            if self.use_argmax:
                action = max(action_probs.items(), key=lambda x: x[1])[0]
            else:
                # Now this won't fail since we ensure action_probs is not empty
                actions, probs = zip(*action_probs.items())
                action = np.random.choice(actions, p=probs)
            
            # Store the observation, policy, and current player for later
            training_examples.append((
                observer.tensor,
                action_probs,
                current_player,
                None  # Placeholder for the reward
            ))
            
            # Apply the action
            state.apply_action(action)
            
            # If chance node after action, resolve it
            while state.is_chance_node():
                outcomes = state.chance_outcomes()
                if outcomes:
                    action = outcomes[0][0]
                    state.apply_action(action)
                else:
                    break
        
        # Game has ended, calculate rewards
        if not state.is_terminal():
            # Game ended due to move limit
            # All examples get negative reward for timeout
            reward = self.forced_terminal_reward
            returns = [reward, -reward]  # Same as state.returns() would be
        else:
            # Game ended normally, get returns
            returns = state.returns()
        
        # Update examples with final rewards
        for i in range(len(training_examples)):
            obs, probs, player, _ = training_examples[i]
            
            # Get reward for this player
            reward = returns[player]
            
            # Apply penalty for taking cards
            if self.take_cards_penalty > 0:
                for a, p in probs.items():
                    if a == ExtraAction.TAKE_CARDS and p > 0:
                        reward -= self.take_cards_penalty
            
            # Apply small penalty per move to encourage faster play
            if self.move_penalty > 0:
                reward -= self.move_penalty
            
            training_examples[i] = (obs, probs, player, reward)
        
        self.game_count += 1
        
        # Format training examples for model training
        formatted_examples = []
        for obs, probs, player, reward in training_examples:
            if reward is not None:  # Skip examples without a reward
                policy_tensor = torch.zeros(self.network.num_actions)
                for action, prob in probs.items():
                    policy_tensor[action] = prob
                
                formatted_examples.append((
                    torch.tensor(obs, dtype=torch.float32),
                    policy_tensor,
                    reward
                ))
        
        return formatted_examples
        
    def play_vs_rule_agent(self, model_player=0):
        """
        Play a game against a rule-based agent and collect training examples.
        The model plays as model_player (0 or 1).
        """
        # Create agents and setup state
        mcts = MCTS(self.network, n_simulations=self.mcts_simulations, 
                   c_puct=self.c_puct, temperature=self.temperature,
                   use_argmax=self.use_argmax, device=self.device)
        
        rule_agent = RuleAgent()
        
        state = self.game.new_initial_state()
        training_examples = []
        move_count = 0
        
        # Handle initial chance node
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            if not outcomes:  # Defensive check
                break
            action = outcomes[0][0]
            state.apply_action(action)
        
        # Main game loop
        while not state.is_terminal() and move_count < (self.max_moves or float('inf')):
            move_count += 1
            
            # Check for legal actions - if none, just end this position
            legal_actions = state.legal_actions()
            if not legal_actions:
                print(f"No legal actions at phase {state._phase}. Ending game state.")
                break
            
            current_player = state.current_player()
            
            if current_player == model_player:
                # Model's turn - get policy via MCTS
                policy = mcts.run(state)
                
                # Handle empty policy case
                if not policy:
                    policy = {action: 1.0/len(legal_actions) for action in legal_actions}
                
                observer = DurakObserver(
                    pyspiel.IIGObservationType(
                        perfect_recall=False, public_info=True,
                        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
                    params=None
                )
                
                observer.set_from(state, current_player)
                
                # Important: Convert to tensor here
                obs_tensor = torch.tensor(observer.tensor, dtype=torch.float32)
                
                # Convert policy to tensor
                policy_tensor = torch.zeros(self.network.num_actions)
                for a, p in policy.items():
                    policy_tensor[a] = p
                
                # Choose action
                if self.use_argmax:
                    action = max(policy.items(), key=lambda x: x[1])[0]
                else:
                    actions, probs = zip(*policy.items())
                    action = np.random.choice(actions, p=probs)
                
                # Store training example with tensor observation and policy
                training_examples.append((
                    obs_tensor,              # Tensor observation
                    policy_tensor,           # Tensor policy
                    current_player,
                    0.0  # Placeholder for reward
                ))
            else:
                # Rule agent's turn - ensure we get a valid action
                try:
                    action = rule_agent.step(state)
                    if not isinstance(action, int) or action not in legal_actions:
                        action = legal_actions[0]  # Fallback
                except Exception:
                    action = legal_actions[0]  # Fallback on any error
            
            # Apply action
            state.apply_action(action)
            
            # Handle chance nodes
            while state.is_chance_node():
                outcomes = state.chance_outcomes()
                if not outcomes:
                    break
                action = outcomes[0][0]
                state.apply_action(action)
        
        # Calculate rewards
        if not state.is_terminal():
            reward = self.forced_terminal_reward
        else:
            returns = state.returns()
            reward = returns[model_player]
        
        # Update examples with final reward
        for i in range(len(training_examples)):
            obs_tensor, policy_tensor, player, _ = training_examples[i]
            value = reward if player == model_player else -reward
            
            # Apply penalties
            if self.move_penalty > 0:
                value -= self.move_penalty * (i + 1) / len(training_examples)
            
            # Store with the final reward
            training_examples[i] = (obs_tensor, policy_tensor, value)
        
        self.game_count += 1
        return training_examples
    
    def _format_training_examples(self, examples):
        """Format training examples for model training."""
        formatted_examples = []
        for obs, probs, player, reward in examples:
            if reward is not None:  # Skip examples without a reward
                policy_tensor = torch.zeros(self.network.num_actions)
                for action, prob in probs.items():
                    policy_tensor[action] = prob
                
                formatted_examples.append((
                    torch.tensor(obs, dtype=torch.float32),
                    policy_tensor,
                    reward
                ))
        
        return formatted_examples
    
    def generate_rule_agent_dataset_for_imitation(self, n_games=500):
        """Generate a dataset of rule agent moves for imitation learning."""
        rule_agent = RuleAgent()
        dataset = []
        
        for _ in tqdm(range(n_games), desc="Generating rule agent dataset"):
            state = self.game.new_initial_state()
            
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
                legal_actions = state.legal_actions()
                if not legal_actions:
                    break
                
                # Get the rule agent's action
                action = rule_agent.step(state)
                
                # Create one-hot encoded action vector
                action_one_hot = np.zeros(self.network.num_actions)
                action_one_hot[action] = 1.0
                
                # Record the observation and action
                observer = DurakObserver(
                    pyspiel.IIGObservationType(
                        perfect_recall=False, public_info=True,
                        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
                    params=None
                )
                observer.set_from(state, current_player)
                
                dataset.append((observer.tensor.copy(), action_one_hot))
                
                # Apply the action
                state.apply_action(action)
        
        return dataset
    
    def train_supervised_on_rule_agent_dataset(self, dataset, batch_size=32, epochs=10):
        """Train the model on rule agent data using supervised learning."""
        self.network.train()
        
        # Convert dataset to tensors
        observations = [torch.tensor(obs, dtype=torch.float32, device=self.device) for obs, _ in dataset]
        actions = [torch.tensor(act, dtype=torch.float32, device=self.device) for _, act in dataset]
        
        dataset_size = len(observations)
        indices = list(range(dataset_size))
        
        for epoch in range(epochs):
            random.shuffle(indices)
            total_loss = 0.0
            num_batches = (dataset_size + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = torch.stack([observations[i] for i in batch_indices])
                batch_actions = torch.stack([actions[i] for i in batch_indices])
                
                self.optimizer.zero_grad()
                policy_logits, _ = self.network(batch_obs)
                
                # Use cross-entropy loss for policy
                loss = F.cross_entropy(policy_logits, torch.argmax(batch_actions, dim=1))
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Supervised Loss: {total_loss/num_batches:.4f}")
    
    def train_step(self, batch):
        """Perform a single training step on a batch of examples."""
        self.optimizer.zero_grad()
        
        # Extract batch components
        observations = [item[0] for item in batch]
        policies = [item[1] for item in batch]
        rewards = [item[2] for item in batch]
        
        # Ensure all are tensors (this is the critical fix)
        observations = [torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs for obs in observations]
        policies = [torch.tensor(pol, dtype=torch.float32) if not isinstance(pol, torch.Tensor) else pol for pol in policies]
        
        # Now stack and move to device
        observations = torch.stack(observations).to(self.device)
        policies = torch.stack(policies).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Forward pass
        policy_logits, values = self.network(observations)
        
        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, policies)
        value_loss = F.mse_loss(values, rewards)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def save_checkpoint(self, iteration):
        """Save a checkpoint of the model."""
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