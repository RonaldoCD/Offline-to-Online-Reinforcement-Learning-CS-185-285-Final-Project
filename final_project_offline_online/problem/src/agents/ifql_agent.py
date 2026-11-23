from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,
 
        discount: float,
        target_update_rate: float,
        flow_steps: int,
        num_samples: int = 3,
        expectile: float = 0.9,
    ):
        super().__init__()

        self.action_dim = action_dim 
        
        # TODO(student): Create flow actor
        self.flow_actor = make_actor_flow(observation_shape, action_dim)

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and value function
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)
        # TODO(student): Create optimizers for all the above models
        self.flow_actor_optimizer = make_actor_flow_optimizer(self.flow_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        """
        # TODO(student): Implement the expectile loss
        loss = abs(expectile - (adv>0).float()) * (adv**2)
        return loss

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        v = self.value(observations)
        q_target = self.target_critic(observations, actions).min(dim=0).values
        loss = self.expectile_loss(v - q_target, self.expectile)
        loss = loss.mean()
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        # TODO(student): Implement rejection sampling
        for n in range(self.num_samples):
            noise = torch.randn((observations.shape[0], self.action_dim), device=observations.device, dtype=observations.dtype)
            actions = self.get_flow_action(observations, noise)
            q_values = self.critic(observations, actions).min(dim=0).values
            if n == 0:
                best_actions = actions
                best_q_values = q_values 
            else:
                mask = q_values > best_q_values
                # print(f"Mask: {mask.sum().item()} / {mask.shape[0]} actions selected")
                best_actions[mask] = actions[mask]
                best_q_values[mask] = q_values[mask]
        return best_actions

    def get_action(self, observation):
        observation = ptu.from_numpy(np.asarray(observation))[None]

        with torch.no_grad():
            action = self.sample_actions(observation)

        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        # TODO(student): Implement euler integration to get flow action
        dt = 1 / self.flow_steps
        action = noise
        for i in range(self.flow_steps):
            t = torch.full(
                (noise.shape[0], 1),
                i * dt,
                device=observation.device,
                dtype=observation.dtype,
            )
            velocity = self.flow_actor(observation, action, t)
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
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        with torch.no_grad():
            y = rewards + self.discount * (1 - dones) * self.value(next_observations)

        q = self.critic(observations, actions)
        loss = ((q - y) ** 2).mean()

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
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        batch_size = observations.shape[0]
        z = torch.randn_like(actions)
        t = torch.rand((batch_size, 1),
                       device=actions.device,
                       dtype=actions.dtype)
        actions_hat = (1 - t) * z + t * actions
        velocities_pred = self.flow_actor(observations, actions_hat, t)
        velocities_target = actions - z
        flow_loss = torch.nn.functional.mse_loss(velocities_pred, velocities_target)

        q_loss = -self.critic(observations, actions_hat).min(dim=0).values.mean()
        loss = flow_loss + q_loss
        self.flow_actor_optimizer.zero_grad()
        loss.backward()
        self.flow_actor_optimizer.step()
        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "q_loss": q_loss,
        }

    # @torch.compile
    # def update_actor(
    #     self,
    #     observations,
    #     actions,
    # ):
    #     batch_size = observations.shape[0]
    #     M = 100
    #     self.alpha = 30
    #     # -------------------------
    #     # IQL advantage weights
    #     # -------------------------
    #     with torch.no_grad():
    #         q = self.critic(
    #             observations,
    #             actions,
    #         ).min(dim=0).values

    #         v = self.value(observations)

    #         advantage = q - v

    #         weights = torch.exp(
    #             self.alpha * advantage
    #         )

    #         weights = torch.clamp(
    #             weights,
    #             max=M,
    #         )

    #     # -------------------------
    #     # Flow matching
    #     # -------------------------
    #     z = torch.randn_like(actions)

    #     t = torch.rand(
    #         (batch_size, 1),
    #         device=actions.device,
    #         dtype=actions.dtype,
    #     )

    #     actions_hat = (1 - t) * z + t * actions

    #     velocities_pred = self.flow_actor(
    #         observations,
    #         actions_hat,
    #         t,
    #     )

    #     velocities_target = actions - z

    #     # Important: keep per-sample losses first.
    #     flow_loss = (
    #         velocities_pred - velocities_target
    #     ).pow(2).mean(dim=-1)

    #     # Advantage-weighted flow matching.
    #     loss = (
    #         weights * flow_loss
    #     ).mean()

    #     self.flow_actor_optimizer.zero_grad()
    #     loss.backward()
    #     self.flow_actor_optimizer.step()

    #     return {
    #         "loss": loss,
    #         "flow_loss": flow_loss.mean(),
    #         "adv_mean": advantage.mean(),
    #         "adv_max": advantage.max(),
    #         "adv_min": advantage.min(),
    #     }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
