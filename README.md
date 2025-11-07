# Deep Monte-Carlo Durak

This repository implements a Monte-Carlo self-play trainer for two-player, 36-card Durak. It replaces the previous AlphaZero/MCTS approach with a model-free Deep Monte-Carlo (DMC) pipeline inspired by DouZero.

## Key features

- Fixed 38-action space with role-aware masking (36 card plays + END_ATTACK + TAKE_CARDS)
- Fast MLP Q-network trained with terminal ±1 Monte-Carlo returns
- ε-greedy self-play with configurable TrueSight (open-hand) warm-up
- Replay buffer and batched MSE updates for high-throughput learning
- Evaluation against a greedy heuristic baseline

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Configuration defaults live in `configs/durak_dmc.yaml`. Launch training with:

```bash
python train_dmc.py
```

Override the config path or device:

```bash
python train_dmc.py --config configs/durak_dmc.yaml --device cpu
```

Checkpoints are saved into `checkpoints/` every configured interval. The training loop prints replay size, running loss, and periodic evaluation win-rate versus the greedy baseline.

### TrueSight warm-up

The `truesight_games` setting enables a short curriculum where both hands are visible in the encoder to jump-start learning. Set it to `0` to disable.

### Scaling tips

- Increase `hidden_scale` to widen the MLP when running on large GPUs.
- Increase `games_per_iter` and `train_steps_per_iter` for higher throughput.
- Reduce the batch size to `512` if you experience memory pressure.

## Evaluation

The helper `eval_vs_greedy` evaluates the learned model against the heuristic greedy player with deterministic argmax selection. Adjust `eval_games` in the config to trade off speed versus statistical confidence.

## Reproducibility

The training script seeds Python, NumPy, and PyTorch. Checkpoint files contain only model weights—store the config separately if you need to resume with identical hyperparameters.
