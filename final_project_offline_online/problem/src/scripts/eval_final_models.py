"""Re-evaluate the final (end-of-training) models of finished runs with more episodes.

Loads each run's agent.pt (saved after the last training step) and its flags.json (config),
runs N evaluation episodes, and writes one row per run to a CSV. No training happens.

Example:
    uv run src/scripts/eval_final_models.py --num_eval_trajectories 100 \
        --runs "exp/s4_fql/sd*_fql_cube-double-*_online_buf150000_wsrl*" \
        --out results/s4_fql_cube-double_buf150k_eval100.csv
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import torch

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu


def evaluate_run(run_dir, n, eval_seed):
    with open(os.path.join(run_dir, "flags.json")) as f:
        flags = json.load(f)
    config = configs.configs[flags["agent"]](flags["env_name"], learning_rate=flags["learning_rate"])
    config["agent_kwargs"].update(flags["agent_kwargs"])  # e.g. alpha used in training

    env = config["make_env_and_dataset"](env_only=True, render=False)
    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    agent = agents[config["agent"]](env.observation_space.shape, env.action_space.shape[-1], **config["agent_kwargs"])
    agent.load_state_dict(torch.load(os.path.join(run_dir, "agent.pt"), map_location=ptu.device))

    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    env.reset(seed=eval_seed)
    trajs = utils.sample_n_trajectories(env, agent, n, ep_len)
    successes = np.array([t["episode_statistics"]["s"] for t in trajs], dtype=float)
    return flags, successes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="globs of run directories")
    p.add_argument("--num_eval_trajectories", type=int, default=100)
    p.add_argument("--eval_seed", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.add_argument("--no_gpu", action="store_true")
    args = p.parse_args()
    ptu.init_gpu(use_gpu=not args.no_gpu)

    dirs = sorted({d for g in args.runs for d in glob.glob(g) if os.path.exists(os.path.join(d, "agent.pt"))})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        flags, s = evaluate_run(d, args.num_eval_trajectories, args.eval_seed)
        m = re.search(r"buf(\d+)_wsrl(\d+)", name)
        row = {
            "run": name, "seed": flags["seed"],
            "initial_offline_steps": int(m.group(1)) if m else "",
            "initial_online_steps": int(m.group(2)) if m else "",
            "final_step": flags["offline_training_steps"] + flags["online_training_steps"],
            "num_episodes": len(s), "success_rate": round(float(s.mean()), 4),
        }
        rows.append(row)
        print(f"{name}: success {s.mean():.2f} over {len(s)} episodes", flush=True)
        with open(args.out, "w", newline="") as f:  # rewrite after each run so partial results are kept
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
