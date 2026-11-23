"""Helpers for starting the online phase: offline-data prefill and reusing an offline checkpoint."""
import csv
import os
from typing import Optional

import torch

from infrastructure.replay_buffer import ReplayBuffer


def prefill_with_offline_data(replay_buffer: ReplayBuffer, dataset: ReplayBuffer, num_transitions: int) -> int:
    """Copy the first `num_transitions` offline transitions into the online replay buffer.

    The first N transitions are taken (not a random subset) so whole trajectories are kept,
    which sequence sampling (action chunking) relies on.
    Returns the number of transitions actually inserted.
    """
    n = min(max(num_transitions, 0), dataset.size)
    if n == 0:
        return 0
    data = dataset.sample_first(n)
    for i in range(n):
        replay_buffer.insert(
            observation=data["observations"][i],
            action=data["actions"][i],
            reward=data["rewards"][i],
            next_observation=data["next_observations"][i],
            done=data["dones"][i],
            mask=data["masks"][i],
        )
    return n


def checkpoint_status(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False).get("status")


def resolve_offline_checkpoint(path: str) -> int:
    """Check `path` is a finished offline checkpoint and return its number of offline steps."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Offline checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("status") != "offline_completed":
        raise ValueError(f"{path} has status {ckpt.get('status')!r}, expected 'offline_completed'")
    return ckpt["next_step"] - 1


def copy_offline_logs(offline_dir: str, train_logger, eval_logger, max_step: int) -> None:
    """Replay the offline run's train/eval CSV rows into the new run's loggers (and wandb),
    so each online-condition run holds the full offline+online curve."""
    for name, logger in (("train.csv", train_logger), ("eval.csv", eval_logger)):
        path = os.path.join(offline_dir, name)
        if not os.path.exists(path):
            print(f"[offline logs] {path} not found, skipping")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(float(row.pop("step")))
                if step > max_step:
                    continue
                values = {k: float(v) for k, v in row.items() if v not in ("", None)}
                logger.log(values, step=step)
