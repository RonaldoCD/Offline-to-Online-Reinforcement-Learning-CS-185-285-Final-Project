"""Train offline ONCE per seed, then fine-tune that same offline model under several online conditions.

Each condition is "OFFLINE_DATA:WSRL_STEPS":
  OFFLINE_DATA -> --initial_offline_steps  (offline transitions copied into the online replay buffer)
  WSRL_STEPS   -> --initial_online_steps   (warm-up env steps without updates, counted in the online budget)

Runs execute one after another (not in parallel). Any argument not listed below is passed through
unchanged to the training script (e.g. --eval_interval 10000 --log_interval 5000).
Arguments for the online runs only go in --online_extra (e.g. --online_extra="--render").

Example:
    uv run src/scripts/run_online_sweep.py --base_config fql \
        --env_name cube-double-play-singletask-task1-v0 --alpha 300 --seeds 0 \
        --run_group s4_fql_double --conditions 0:0 1000000:0 100000:0 0:5000 0:20000 \
        --eval_interval 10000

Layout:
    exp/<run_group>/sd0_fql_<env>_a300.0_offline/checkpoint_500000.pt     <- offline model (trained once)
    exp/<run_group>/sd0_fql_<env>_a300.0_online_buf0_wsrl5000/eval.csv    <- offline curve + online curve
"""
import argparse
import os
import shlex
import subprocess
import sys

import torch


def parse_condition(text):
    try:
        buf, wsrl = (int(x) for x in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"condition must look like OFFLINE_DATA:WSRL_STEPS, got {text!r}")
    return buf, wsrl


def offline_done(path):
    if not os.path.exists(path):
        return False
    return torch.load(path, map_location="cpu", weights_only=False).get("status") == "offline_completed"


def run(cmd, dry_run):
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", default="train_offline_online",
                   choices=["train_offline_online", "train_ac_offline_online"])
    p.add_argument("--base_config", required=True)
    p.add_argument("--env_name", required=True)
    p.add_argument("--run_group", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--offline_training_steps", type=int, default=500000)
    p.add_argument("--online_training_steps", type=int, default=100000)
    p.add_argument("--conditions", type=parse_condition, nargs="+", required=True,
                   help="OFFLINE_DATA:WSRL_STEPS pairs, e.g. 0:0 1000000:0 0:5000")
    p.add_argument("--name_suffix", default="",
                   help="Extra tag for run names when passing other hyperparameters through (e.g. _e0.9)")
    p.add_argument("--online_extra", default="",
                   help='Extra args for the online runs only, e.g. --online_extra="--render"')
    p.add_argument("--dry_run", action="store_true", help="Only print the commands")
    args, passthrough = p.parse_known_args()

    for buf, wsrl in args.conditions:
        if wsrl > args.online_training_steps:
            p.error(f"WSRL steps ({wsrl}) exceed the online budget ({args.online_training_steps})")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.script}.py")
    common = [sys.executable, script,
              "--base_config", args.base_config,
              "--env_name", args.env_name,
              "--run_group", args.run_group]
    if args.alpha is not None:
        common += ["--alpha", str(args.alpha)]
    common += passthrough

    for seed in args.seeds:
        base = f"sd{seed}_{args.base_config}_{args.env_name}"
        if args.alpha is not None:
            base += f"_a{args.alpha}"
        base += args.name_suffix

        # 1) Offline phase, once per seed (resumes automatically if interrupted)
        offline_name = f"{base}_offline"
        ckpt = os.path.join("exp", args.run_group, offline_name, f"checkpoint_{args.offline_training_steps}.pt")
        if offline_done(ckpt):
            print(f"\n[seed {seed}] reusing offline model {ckpt}")
        else:
            run(common + ["--seed", str(seed), "--exp_name", offline_name,
                          "--offline_training_steps", str(args.offline_training_steps),
                          "--online_training_steps", "0"], args.dry_run)

        # 2) Online fine-tuning from that model, one condition at a time
        for buf, wsrl in args.conditions:
            run(common + ["--seed", str(seed), "--exp_name", f"{base}_online_buf{buf}_wsrl{wsrl}",
                          "--offline_checkpoint", ckpt,
                          "--online_training_steps", str(args.online_training_steps),
                          "--initial_offline_steps", str(buf),
                          "--initial_online_steps", str(wsrl)] + shlex.split(args.online_extra), args.dry_run)


if __name__ == "__main__":
    main()
