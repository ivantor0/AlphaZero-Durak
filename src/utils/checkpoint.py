# src/utils/checkpoint.py
import os
import torch
import warnings

def save_checkpoint(network, game_count, optimizer=None, iteration=None, checkpoint_dir="checkpoints"):
    """Save a checkpoint of the model."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_name = f"{game_count}.ckpt" if iteration is None else f"{iteration}.ckpt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Save architecture information
    checkpoint = {
        'model_state_dict': network.state_dict(),
        'game_count': game_count,
        'model_config': {
            'input_dim': network.input_dim,
            'hidden_dim': network.hidden_dim, 
            'num_actions': network.num_actions,
            'num_layers': len(network.hidden_layers),
            'use_history': getattr(network, 'use_history', False),
            'history_dim': getattr(network, 'history_dim', None)
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if iteration is not None:
        checkpoint['iteration'] = iteration
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, network, optimizer=None, device="cpu"):
    """Load a checkpoint."""
    # Use weights_only=False but handle the warning
    warnings.filterwarnings("ignore", category=FutureWarning, 
                           module="torch.serialization")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if model configurations match
    if 'model_config' in checkpoint:
        saved_config = checkpoint['model_config']
        current_config = {
            'input_dim': network.input_dim,
            'hidden_dim': network.hidden_dim, 
            'num_actions': network.num_actions,
            'num_layers': len(network.hidden_layers),
            'use_history': getattr(network, 'use_history', False),
            'history_dim': getattr(network, 'history_dim', None)
        }
        
        # If configurations don't match, provide a useful error message
        if saved_config != current_config:
            print("Warning: Model configuration mismatch!")
            print("Saved model configuration:")
            for k, v in saved_config.items():
                print(f"  {k}: {v}")
            print("Current model configuration:")
            for k, v in current_config.items():
                print(f"  {k}: {v}")
            
            # Special handling for use_history mismatch
            if saved_config.get('use_history') != current_config.get('use_history'):
                print("Critical mismatch: 'use_history' differs between saved and current model!")
                if not current_config.get('use_history') and saved_config.get('use_history'):
                    print("Cannot load a history-based model into a non-history model.")
                    print("Try initializing your network with use_history=True")
                    return 0, None
    
    # Use strict=False to ignore missing/unexpected keys
    network.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Print any unloaded parameters
    current_params = set(network.state_dict().keys())
    loaded_params = set(checkpoint['model_state_dict'].keys())
    
    if current_params != loaded_params:
        if current_params - loaded_params:
            print("Parameters in current model but not in checkpoint:")
            for param in current_params - loaded_params:
                print(f"  {param}")
        
        if loaded_params - current_params:
            print("Parameters in checkpoint but not used in current model:")
            for param in loaded_params - current_params:
                print(f"  {param}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    game_count = checkpoint.get('game_count', 0)
    iteration = checkpoint.get('iteration', None)
    
    print(f"Loaded checkpoint from {checkpoint_path}" + 
          (f" (iteration {iteration})" if iteration is not None else ""))
    
    return game_count, iteration