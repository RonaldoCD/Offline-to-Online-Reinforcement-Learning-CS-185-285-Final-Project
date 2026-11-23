from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class QCFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
        horizon: int,
        num_samples: int = 1,
        q_agg: str = "mean",
        num_critics: int = 2,
    ):
        super().__init__()
 
        self.action_dim = action_dim
        self.horizon = horizon
        self.chunk_size = self.action_dim * self.horizon
        print(f"QCFQLAgent: action_dim={self.action_dim}, horizon={self.horizon}, chunk_size={self.chunk_size}")
        self.q_agg = q_agg
        
        self.bc_actor = make_bc_actor(observation_shape, self.chunk_size)
        self.onestep_actor = make_onestep_actor(observation_shape, self.chunk_size)
        self.critic = make_critic(observation_shape, self.chunk_size, num_critics)
        self.target_critic = make_critic(observation_shape, self.chunk_size, num_critics)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha
        # self.register_buffer(
        #     "alpha",
        #     torch.tensor(float(alpha), dtype=torch.float32)
        # )
        self.num_samples = num_samples

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
            1. Sample multiple candidate actions via the BC flow.
            2. Evaluate them with the critic.
            3. Pick the action with the highest Q-value.
        """
        # print("Sampling actions...")
        for n in range(self.num_samples):
            z = torch.randn((observations.shape[0], self.chunk_size), device=observations.device, dtype=observations.dtype)
            actions = self.onestep_actor(observations, z)
            if self.q_agg == 'mean':
                q_values = self.critic(observations, actions).mean(dim=0)
            else:
                q_values = self.critic(observations, actions)
                q_values = q_values.min(dim=0).values
                
            if n == 0:
                best_actions = actions
                best_q_values = q_values 
            else:
                mask = q_values > best_q_values
                # print(f"Mask: {mask.sum().item()} / {mask.shape[0]} actions selected")
                best_actions[mask] = actions[mask]
                best_q_values[mask] = q_values[mask]
        best_actions = torch.clamp(best_actions, -1, 1)
        return best_actions


    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        with torch.no_grad():
            action = self.sample_actions(observation)
        
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        dt = 1 / self.flow_steps
        action = noise
        for i in range(self.flow_steps):
            t = torch.full(
                (noise.shape[0], 1),
                i * dt,
                device=observation.device,
                dtype=observation.dtype,
            )
            velocity = self.bc_actor(observation, action, t)
            action = action + dt * velocity
        action = torch.clamp(action, -1, 1)
        return action


    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        valids: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, -1)
        with torch.no_grad():
            z = torch.randn_like(actions)
            next_actions = torch.clamp(self.onestep_actor(next_observations, z), -1, 1) ## ToFix: Check next_actions shape
            # print(f"next_actions shape: {next_actions.shape}, next_observations shape: {next_observations.shape}")
            if self.q_agg == 'mean':
                q_next = self.target_critic(next_observations, next_actions).mean(dim=0)                
            else:
                q_next = self.target_critic(next_observations, next_actions)
                q_next = q_next.min(dim=0).values
                
            y = rewards[:, -1] + (self.discount**self.horizon) * masks[:,-1] * q_next 

        q = self.critic(observations, actions)
        # print(f"q shape: {q.shape}, y shape: {y.shape}, valids shape: {valids.shape}, actions shape: {actions.shape}" )
        # As in QC (acfql.py): drop chunks whose last action belongs to another trajectory.
        # `valids` is built from `dones` = trajectory ends (dataset `terminals`), NOT from
        # success, so chunks that reach the goal mid-chunk are kept.
        loss = (((q - y.unsqueeze(0)) ** 2) * valids[:, -1]).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        valids: torch.Tensor
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, -1)

        z = torch.randn_like(actions)
        t = torch.rand((batch_size, 1),
                       device=actions.device,
                       dtype=actions.dtype)
        
        actions_hat = (1 - t) * z + t * actions
        velocities_pred = self.bc_actor(observations, actions_hat, t)
        velocities_target = actions - z
        flow_error = (velocities_pred - velocities_target)**2
        flow_error = flow_error.reshape(batch_size, self.horizon, self.action_dim)
        loss = torch.mean(flow_error * valids[..., None])

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        # valids: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, -1)

        z = torch.randn_like(actions)

        actions_actor = self.onestep_actor(observations, z)

        with torch.no_grad():
            actions_bc_actor = self.get_bc_action(observations, z)

        mse = torch.mean((actions_actor - actions_bc_actor)**2)
        distill_loss = self.alpha * mse

        # Additional metrics for logging.
        
        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        actions_actor = torch.clamp(actions_actor, -1, 1)
        q_loss = - self.critic(observations, actions_actor).mean() 

        # Total loss.
        loss = distill_loss + q_loss

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        valids: torch.Tensor,
        masks: torch.Tensor,
        step: int,
        offline: bool = True,
    ):
        # self.set_alpha(step, offline=offline)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones, valids, masks)
        
        metrics_bc_actor = self.update_bc_actor(observations, actions, valids)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)

    # @torch.no_grad()
    # def set_alpha(
    #     self,
    #     step: int,
    #     total_steps: int = 500000,
    #     start_value : float = 600.0,
    #     end_value: float = 100.0,
    #     offline: bool = True,
    # ):
    #     if offline:
    #         progress = min(step / total_steps, 1.0)
    #         self.alpha.fill_(start_value + progress * (end_value - start_value))
    #     else:
    #         self.alpha.fill_(end_value)
        # print(f"Step {step}: alpha set to {self.alpha:.4f} )")