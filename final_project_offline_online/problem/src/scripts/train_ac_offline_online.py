import argparse
import os
import tempfile
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer
from collections import deque

import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.online_init import (
    prefill_with_offline_data,
    checkpoint_status,
    resolve_offline_checkpoint,
    copy_offline_logs,
)

def get_run_name(args: argparse.Namespace) -> str:
    if args.exp_name is not None:
        return args.exp_name

    exp_name = f"sd{args.seed}_{args.base_config}_{args.env_name}"
    if args.alpha is not None:
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.expectile is not None:
        exp_name = f"{exp_name}_e{args.expectile}"
    return exp_name


def load_checkpoint(
    save_dir: str,
    agent=None,
    start_online_training: bool = False,
    checkpoint_path: Optional[str] = None,
) -> tuple[int, bool]:

    # ------------------------------------------------------------
    # Select checkpoint
    # ------------------------------------------------------------
    if start_online_training and checkpoint_path is not None:
        # Reuse an offline model trained in another run (see --offline_checkpoint)
        offline_checkpoint_path = checkpoint_path
    elif start_online_training:
        offline_checkpoint_path = None

        for filename in os.listdir(save_dir):
            if not filename.startswith("checkpoint_") or not filename.endswith(".pt"):
                continue

            path = os.path.join(save_dir, filename)

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            if checkpoint.get("status") == "offline_completed":
                offline_checkpoint_path = path
                break

        if offline_checkpoint_path is None:
            return 0, False

        checkpoint_path = offline_checkpoint_path

    else:
        checkpoint_path = os.path.join(save_dir, "checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            return 0, False

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Only apply this logic when normally resuming training.
    #
    # If start_online_training=True, we deliberately want to load
    # the offline_completed checkpoint.
    if not start_online_training:
        if checkpoint.get("status") == "completed":
            return 0, True

    # ------------------------------------------------------------
    # Only retrieve step
    # ------------------------------------------------------------
    if agent is None:
        return checkpoint["next_step"], False

    # ------------------------------------------------------------
    # Restore agent
    # ------------------------------------------------------------
    agent.load_state_dict(checkpoint["agent_state_dict"])

    # ------------------------------------------------------------
    # Restore optimizers
    # ------------------------------------------------------------
    for name, state_dict in checkpoint["optimizer_states"].items():
        optimizer = getattr(agent, name, None)

        if isinstance(optimizer, Optimizer):
            optimizer.load_state_dict(state_dict)

            # Move optimizer tensors to current device
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(ptu.device)

    # ------------------------------------------------------------
    # Restore RNG states
    # ------------------------------------------------------------
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

    if (
        torch.cuda.is_available()
        and "torch_cuda_rng_state" in checkpoint
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["torch_cuda_rng_state"]
        )

    return checkpoint["next_step"], False

def save_checkpoint(
    save_dir: str,
    agent,
    next_step: int,
    status: str,
    checkpoint_callback: Optional[Callable[[], None]] = None,
) -> None:
    optimizer_states = {}
    for name, value in vars(agent).items():
        if isinstance(value, Optimizer):
            optimizer_states[name] = value.state_dict()

    checkpoint = {
        "next_step": next_step,
        "status": status,
        "agent_state_dict": agent.state_dict(),
        "optimizer_states": optimizer_states,
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()

    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    with tempfile.NamedTemporaryFile(dir=save_dir, suffix=".tmp", delete=False) as f:
        temp_path = f.name
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    # ------------------------------------------------------------
    # Preserve offline checkpoint
    # ------------------------------------------------------------
    if status == "offline_completed":
        offline_checkpoint_path = os.path.join(
            save_dir,
            f"checkpoint_{next_step-1}.pt",
        )
        torch.save(checkpoint, offline_checkpoint_path)

    if checkpoint_callback is not None:
        checkpoint_callback()



def run_offline_training_loop(
    config: dict, 
    train_logger, 
    eval_logger, 
    args: argparse.Namespace, 
    start_step: int = 0,
    checkpoint_callback: Optional[Callable[[], None]] = None,
    ):
    """
    Run offline training loop
    """
    # TODO(student): Implement offline training loop
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    config["training_phase"] = "offline_training"
    
    # Make the gymnasium environment
    env, dataset = config["make_env_and_dataset"]()
    render_env = config["make_env_and_dataset"](env_only=True, render=args.render)

    example_batch = dataset.sample(1)
    print("Config: ", config)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch['observations'].shape[1:],
        example_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )
    # dataset.capacity = dataset.capacity + 5000 * 6 
    ## Check
    start_step, is_completed = load_checkpoint(args.save_dir, agent)

    print("Start step: ", start_step)
    print("is completed", is_completed)
    if is_completed:
        return

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    step_iterator = range(start_step, config["offline_training_steps"] + 1)
    if start_step > 0:
        remaining_steps = config["offline_training_steps"] - start_step + 1
        print(f"Offline training Resuming from step {start_step} with {remaining_steps} steps remaining.")
    progress_bar = tqdm.tqdm(
        step_iterator,
        total=config["offline_training_steps"] + 1,
        initial=start_step,
        dynamic_ncols=True,
    )

    for step in progress_bar:
        ## Train with offline RL
        batch = dataset.sample_sequence(config["batch_size"], sequence_length=config["agent_kwargs"]["horizon"], discount=config["agent_kwargs"]["discount"])

        # print("Observations: ", batch["observations"].shape)
        # print("Actions: ", batch["actions"].shape)
        # print("Rewards: ", batch["rewards"].shape)
        # print("Next Observations: ", batch["next_observations"].shape)
        # print("Dones: ", batch["dones"].shape)
        # print("Valids: ", batch["valids"].shape)

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            batch["valids"],
            batch["masks"],
            step,
            # offline=True
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:  # includes step 0 (untrained policy)
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                env, 
                agent,
                args.num_eval_trajectories,
                ep_len
            )

            successes = [t["episode_statistics"]["s"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/success_rate": float(np.mean(successes))
                },
                step = step
            )

            save_checkpoint(
                args.save_dir,
                agent,
                next_step=step + 1,
                status="offline_running",
                checkpoint_callback=checkpoint_callback,
            )
            
            if args.num_render_trajectories > 0 and args.render:
                video_trajectories = utils.sample_n_trajectories(
                    render_env, 
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=args.render,
                )

                eval_logger.log_paths_as_videos(
                    video_trajectories,
                    step=step,
                    fps = 10,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts"
                )

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    save_checkpoint(
                    args.save_dir,
                    agent,
                    next_step=config["offline_training_steps"] + 1,
                    status="offline_completed",
                    checkpoint_callback=checkpoint_callback,
                    )        


def run_online_training_loop(
        config: dict, 
        train_logger, 
        eval_logger, 
        args: argparse.Namespace, 
        start_step: int = 0,
        checkpoint_callback: Optional[Callable[[], None]] = None,):
    """
    Run online training loop
    """
    # TODO(student): Implement online training loop
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    config["training_phase"] = "online_training"

    env, dataset = config["make_env_and_dataset"]()
    eval_env = config["make_env_and_dataset"](env_only=True, render=False)
    render_env = config["make_env_and_dataset"](env_only=True, render=args.render)
    print("Dataset length: ", len(dataset.observations))
        
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[-1]
    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        obs_shape,
        action_dim,
        **config["agent_kwargs"],
    )
    start_online_training = False
    if args.offline_training_steps == start_step:
        start_online_training = True
        print("Starting online training after offline training completed.")

    ## Check
    start_step, is_completed = load_checkpoint(
        args.save_dir, agent,
        start_online_training=start_online_training,
        checkpoint_path=args.offline_checkpoint,
    )

    if is_completed:
        return

    # Offline data retention: copy the first `initial_offline_steps` offline transitions
    # into the online replay buffer (made big enough to never overwrite them).
    capacity = max(config["replay_buffer_capacity"], args.initial_offline_steps + config["online_training_steps"])
    replay_buffer = ReplayBuffer(capacity)
    n_prefill = prefill_with_offline_data(replay_buffer, dataset, args.initial_offline_steps)
    print(f"Online replay buffer prefilled with {n_prefill} offline transitions")

    # WSRL warm-up: the first `initial_online_steps` online env steps only collect data with the
    # offline policy (no gradient updates). They count towards online_training_steps.
    first_update_step = start_step + args.initial_online_steps
    print(f"WSRL warm-up: {args.initial_online_steps} steps, first update at step {first_update_step}")

    observation, info = env.reset()

    step_iterator = range(start_step, config["online_training_steps"] + config["offline_training_steps"] + 1)
    if start_step > config["offline_training_steps"]:
        remaining_steps = config["online_training_steps"] + config["offline_training_steps"] - start_step + 1
        print(f"Resuming from step {start_step} with {remaining_steps} steps remaining.")

    progress_bar = tqdm.tqdm(
        step_iterator,
        total=config["offline_training_steps"] + config["online_training_steps"] + 1,
        initial=start_step,
        dynamic_ncols=True,
    )

    action_queue = deque()
            
    for step in progress_bar:

        if len(action_queue) == 0:
            action_chunk = agent.get_action(observation)
            action_chunk = np.asarray(action_chunk).reshape(-1, action_dim,)
            action_queue.extend(action_chunk)

        action = action_queue.popleft()
        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        mask = 1.0 - float(terminated)

        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            mask=mask
        )

        if done:
            observation, info = env.reset()
            action_queue.clear()
        else:
            observation = next_observation

        # Train the agent
        # Update only after the WSRL warm-up and once the buffer holds a full batch
        if step >= first_update_step and replay_buffer.size >= config["batch_size"]:
            # TODO(Section 3.1): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample_sequence(config["batch_size"], sequence_length=config["agent_kwargs"]["horizon"], discount=config["agent_kwargs"]["discount"])

            batch = ptu.from_numpy(batch)

            metrics = agent.update(
                        batch["observations"],
                        batch["actions"],
                        batch["rewards"],
                        batch["next_observations"],
                        batch["dones"],
                        batch["valids"],
                        batch["masks"],
                        step,
                        # offline=False
                    )

            if step % args.log_interval == 0:
                train_logger.log(metrics, step=step)
            
        # Evaluate (also during the WSRL warm-up)
        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                            eval_env, 
                            agent,
                            args.num_eval_trajectories,
                            ep_len
                        )

            successes = [t["episode_statistics"]["s"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/success_rate": float(np.mean(successes))
                },
                step = step
            )

            save_checkpoint(
                args.save_dir,
                agent,
                next_step=step + 1,
                status="online_running",
                checkpoint_callback=checkpoint_callback,
            )

            if args.num_render_trajectories > 0 and args.render:
                video_trajectories = utils.sample_n_trajectories(
                    render_env, 
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=args.render,
                )

                eval_logger.log_paths_as_videos(
                    video_trajectories,
                    step=step,
                    fps = 10,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts"
                )

    dump_log(agent, train_logger, eval_logger, config, args.save_dir)
    save_checkpoint(
                    args.save_dir,
                    agent,
                    next_step=config["offline_training_steps"] + 1,
                    status="completed",
                    checkpoint_callback=checkpoint_callback,
                )

def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    
    # Online retention of offline data
    # TODO(student): If desired, add arguments for online retention of offline data
    parser.add_argument("--initial_offline_steps", type=int, default=0,
                        help="Offline-data retention: number of offline-dataset transitions copied into the "
                             "online replay buffer before online training starts")
    parser.add_argument("--offline_checkpoint", type=str, default=None,
                        help="Path to an 'offline_completed' checkpoint (e.g. exp/<group>/<run>/checkpoint_500000.pt). "
                             "Skips offline training and fine-tunes this model online")
    parser.add_argument("--render", action="store_true", help="Render evaluation trajectories")
    parser.add_argument("--num_render_trajectories", type=int, default=4)  # Number of offline samples to use for online training
    parser.add_argument("--lr", type=float, default=3e-4)  # Learning rate\
    parser.add_argument("--num_critics", type=int, default=2)  # Number of critics for QCFQL
    parser.add_argument("--q_agg", type=str, default="mean")  # Q aggregation method for QCFQL
    # WSRL
    parser.add_argument("--initial_online_steps", type=int, default=0,
                        help="WSRL warm-up: online env steps collected with the offline policy before any "
                             "gradient update. Counted within --online_training_steps")
    
 
    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    ## QCFQL
    parser.add_argument("--horizon", type=int, default=5)

    ## QAM
    parser.add_argument("--flow_steps", type=int, default=10)

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config = configs.configs[args.base_config](args.env_name,
                                               learning_rate=args.lr,
                                               )

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    config['horizon'] = args.horizon
    config['flow_steps'] = args.flow_steps
    # TODO(student): If necessary, add additional config values

    exp_name = f"sd{args.seed}_{config['log_name']}"
    # exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.num_critics != 2:
        config['agent_kwargs']['num_critics'] = args.num_critics
        exp_name = f"{exp_name}_q{args.num_critics}"
    if args.horizon != 5:
        config['agent_kwargs']['horizon'] = args.horizon
        exp_name = f"{exp_name}_h{args.horizon}"
    if args.online_training_steps > 0 and args.initial_offline_steps > 0:
        exp_name = f"{exp_name}_buf{args.initial_offline_steps}"
    if args.online_training_steps > 0 and args.initial_online_steps > 0:
        exp_name = f"{exp_name}_wsrl{args.initial_online_steps}"
    if args.lr != 3e-4:
        exp_name = f"{exp_name}_lr{args.lr}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"
    # if args.q_agg is not None:
    #     # config['agent_kwargs']['q_agg'] = args.q_agg
    #     exp_name = f"{exp_name}_{args.q_agg}"
        
    if args.exp_name is not None:
        exp_name = args.exp_name  # explicit run name (used by run_online_sweep.py)
    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    if args.offline_checkpoint is not None:
        # Reuse an offline model: skip offline training, start online fine-tuning from it
        if checkpoint_status(os.path.join(args.save_dir, "checkpoint.pt")) == "completed":
            print(f"{args.save_dir} already completed, nothing to do.")
            return
        offline_steps = resolve_offline_checkpoint(args.offline_checkpoint)
        args.offline_training_steps = offline_steps
        config['offline_training_steps'] = offline_steps
        print(f"Reusing offline checkpoint {args.offline_checkpoint} ({offline_steps} offline steps)")
        copy_offline_logs(os.path.dirname(args.offline_checkpoint), train_logger, eval_logger, offline_steps)
        start_step = offline_steps
    elif args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        # TODO(student): Implement offline training loop
        # Hint: You might consider passing the agent's path to the online training loop
        run_offline_training_loop(config, train_logger, eval_logger, args, start_step)
        start_step = args.offline_training_steps
        
    
    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
        # TODO(student): Implement online training loop
        run_online_training_loop(config, train_logger, eval_logger, args, start_step)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
