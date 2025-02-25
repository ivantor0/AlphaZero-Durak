# src/utils/checkpoint.py
import os
import torch

def save_checkpoint(network, game_count, optimizer=None, iteration=None, checkpoint_dir="checkpoints"):
    """Save a checkpoint of the model."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_name = f"{game_count}.ckpt" if iteration is None else f"{iteration}.ckpt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    checkpoint = {
        'model_state_dict': network.state_dict(),
        'game_count': game_count
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if iteration is not None:
        checkpoint['iteration'] = iteration
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, network, optimizer=None, device="cpu"):
    """Load a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    game_count = checkpoint.get('game_count', 0)
    iteration = checkpoint.get('iteration', None)
    
    print(f"Loaded checkpoint from {checkpoint_path}" + 
          (f" (iteration {iteration})" if iteration is not None else ""))
    
    return game_count, iteration