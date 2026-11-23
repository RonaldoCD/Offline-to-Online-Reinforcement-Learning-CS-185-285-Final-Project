from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_noise_critic,
        make_noise_critic_optimizer,
        make_alpha,
        make_alpha_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        # TODO(student): Create BC flow actor and target BC flow actor
        # TODO(student): Create noise policy
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and z critic (for noise policy)
        # TODO(student): Create learnable entropy coefficient
        # TODO(student): Create optimizers for all the above models
                        
        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())
        self.noise_actor = make_noise_actor(observation_shape, action_dim)

        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.z_critic = make_noise_critic(observation_shape, action_dim)
        # self.target_z_critic = make_z_critic(observation_shape, action_dim)
        # self.target_z_critic.load_state_dict(self.z_critic.state_dict())

        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_noise_critic_optimizer(self.z_critic.parameters())

        self.log_alpha = make_alpha()
        self.log_alpha_optimizer = make_alpha_optimizer(self.log_alpha.parameters())
        self.to(ptu.device)

    @property
    def alpha(self):
        # TODO(student): Allow access to the learnable entropy coefficient (tip: if you are learning log alpha, as in HW3, then when we want to use alpha, you should return the exponential of the log alpha)
        # LogParam.forward() already returns exp(log_param), i.e. alpha itself
        return self.log_alpha()

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        # TODO(student): Implement Euler integration of BC flow. Keep in mind that the target BC flow actor should be used
        # Also note that we can control what we use as the noise input (could be sampled from a noise policy or from a normal distribution)
        dt = 1 / self.flow_steps
        actions = noises
        for i in range(self.flow_steps):
            t = torch.full(
                        (noises.shape[0], 1),
                        i * dt,
                        device=observations.device,
                        dtype=observations.dtype,
                    )
            velocity = self.target_bc_flow_actor(observations, actions, t)
            actions = actions + dt * velocity
        actions = torch.clamp(actions, -1, 1)
        return actions
            

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        # TODO(student): Sample noise from the noise policy and use to sample actions from the BC flow policy
        noises_dist = self.noise_actor(observations)
        noises = noises_dist.sample()
        actions = self.sample_flow_actions(observations, self.noise_scale * noises)
        return actions
    
    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        # TODO(student): Implement get action
        observation = ptu.from_numpy(np.asarray(observation))[None]
        
        with torch.no_grad():
            actions = self.sample_actions(observation)

        return ptu.to_numpy(actions)[0]
        
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        # TODO(student): Implement critic loss
        
        with torch.no_grad():
            z_dist = self.noise_actor(next_observations)
            z = z_dist.sample()
            next_actions = self.sample_flow_actions(next_observations, self.noise_scale * z)
            q_target_next = self.target_critic(next_observations, next_actions)
            y = rewards + self.discount * (1 - dones) * q_target_next.mean(dim=0)

        q = self.critic(observations, actions)
        loss = ((y - q)**2).mean()  

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        # TODO(student): Update critic

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
    
    def update_qz(self, 
        observations: torch.Tensor,
        actions: torch.Tensor,
        # noises: torch.Tensor,
    ) -> dict:
        """Update z_critic."""
        noises = torch.randn_like(actions)

        # TODO(student): Implement z_critic loss
        with torch.no_grad():
            actions_bc = self.sample_flow_actions(observations, noises)
            y = self.critic(observations, actions_bc).mean(dim=0)

        q_z = self.z_critic(observations, noises)
        loss = ((q_z - y)**2).mean()

        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()
        # TODO(student): Update z_critic
        
        return {
            "q_z_loss": loss,
            "q_z_mean": q_z.mean(),
            "q_z_max": q_z.max(),
            "q_z_min": q_z.min(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        # TODO(student): Implement BC flow loss
        batch_size = observations.shape[0]

        z = torch.randn_like(actions)
        t = torch.rand(
            (batch_size, 1),
            device=actions.device,
            dtype=actions.dtype,
        )

        actions_hat = (1 - t) * z + t * actions

        velocities_pred = self.bc_flow_actor(observations, actions_hat, t)
        with torch.no_grad():
            velocities_target = actions - z

        loss = nn.functional.mse_loss(velocities_pred, velocities_target)

        # q_loss = -self.critic(observations, actions_hat).min(dim=0).values.mean()
        # loss = flow_loss + q_loss
    
        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()

        # TODO(student): Update BC flow actor
        
        return {
            "loss": loss,
            # "q_loss": q_loss,
            # "flow_loss": flow_loss,
        }
    
    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        # TODO(student): Implement noise actor loss
        z_dist = self.noise_actor(observations)
        z = z_dist.rsample()
        log_probs = z_dist.log_prob(z)
        q_loss = - self.z_critic(observations, self.noise_scale * z).mean()
        h_loss = self.alpha * log_probs.mean()
        loss = q_loss + h_loss

        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()
        # TODO(student): Update noise actor
        
        return {
            "loss": loss,
            "q_loss": q_loss,
            "h_loss": h_loss,
        }

    def update_alpha(self, observations) -> dict:
        """Update alpha."""
        # TODO(student): Implement alpha loss
        z_dist = self.noise_actor(observations)
        z = z_dist.rsample()
        log_probs = z_dist.log_prob(z)
        entropy_target = - self.action_dim
        entropy_estimate = - log_probs.mean()

        loss = self.log_alpha() * (entropy_estimate - entropy_target).detach()

        # TODO(student): Update alpha
        self.log_alpha_optimizer.zero_grad()
        loss.backward()
        self.log_alpha_optimizer.step()
                
        return {
            "loss": loss,
            "entropy": entropy_estimate,
        }

    def update_target_critic(self) -> None:
        # TODO(student): Implement target critic update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
        
    def update_target_bc_flow_actor(self) -> None:
        # TODO(student): Implement target BC flow actor update
        for target_param, param in zip(self.target_bc_flow_actor.parameters(), self.bc_flow_actor.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
        
    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        # TODO(student): Update critic, z_critic, actor, noise actor, and alpha - feel free to modify this code according to your setup!
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations, actions)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha(observations)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
            **{f"critic_noise/{k}": v.item() for k, v in metrics_qz.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics

