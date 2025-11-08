from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import yaml

from src.agents.qnet import QNet
from src.durak.durak_game import initial_state
from src.durak.encoding import encode_state
from src.evals.eval_vs_greedy import eval_vs_greedy
from src.rl.trainer import DMCTrainer, TrainerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_input_dim(sample_state) -> int:
    obs = encode_state(sample_state, perspective_player=0, truesight=True)
    return int(obs.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Monte-Carlo training for Durak")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/durak_dmc.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    if args.device is not None:
        cfg_dict["device"] = args.device

    set_seed(cfg_dict["seed"])
    device_str = cfg_dict.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    rng = np.random.default_rng(cfg_dict["seed"])
    init_state = initial_state(rng)
    input_dim = _get_input_dim(init_state)

    qnet = QNet(input_dim, n_actions=38, hidden_scale=cfg_dict.get("hidden_scale", 1.0))
    trainer_cfg = TrainerConfig(
        lr=cfg_dict["lr"],
        replay_size=cfg_dict["replay_size"],
        batch_size=cfg_dict["batch_size"],
        grad_clip=cfg_dict["grad_clip"],
        games_per_iter=cfg_dict["games_per_iter"],
        train_steps_per_iter=cfg_dict["train_steps_per_iter"],
    )
    trainer = DMCTrainer(qnet, device, trainer_cfg)

    total_games = 0
    eps = cfg_dict["eps_start"]
    decay_games = cfg_dict["eps_decay_games"]
    truesight_games = cfg_dict.get("truesight_games", 0)
    eval_interval = cfg_dict.get("eval_interval_iters", 5)
    eval_games = cfg_dict.get("eval_games", 200)
    checkpoint_interval = cfg_dict.get("checkpoint_interval_iters", 5)

    iteration = 0
    while True:
        truesight = total_games < truesight_games
        wins, episodes = trainer.selfplay_and_fill(
            cfg_dict["games_per_iter"], eps, truesight, rng
        )
        total_games += episodes
        iteration += 1

        if total_games < decay_games:
            frac = total_games / decay_games
            eps = cfg_dict["eps_start"] + (cfg_dict["eps_end"] - cfg_dict["eps_start"]) * frac
        else:
            eps = cfg_dict["eps_end"]

        losses = []
        for _ in range(cfg_dict["train_steps_per_iter"]):
            loss = trainer.train_step()
            if loss is not None:
                losses.append(loss)

        avg_loss = float(np.mean(losses)) if losses else None
        print(
            f"[Iter {iteration}] games={total_games} eps={eps:.3f} replay={len(trainer.replay)} "
            f"loss={avg_loss} wins_as_p0={wins}/{episodes}",
            flush=True,
        )

        if iteration % eval_interval == 0:
            eval_wr = eval_vs_greedy(qnet, device, n_games=eval_games)
            print(f"  Eval vs greedy win rate: {eval_wr:.3f}", flush=True)

        if iteration % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"qnet_iter{iteration}.pt")
            torch.save(qnet.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
