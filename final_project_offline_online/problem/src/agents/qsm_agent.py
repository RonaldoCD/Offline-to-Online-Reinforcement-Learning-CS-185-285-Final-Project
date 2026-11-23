from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float, 
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create actor
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # TODO(student): Create optimizers for all the above models
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps

        betas = self.cosine_beta_schedule(flow_steps)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(1 - betas, dtype=torch.float32))
        self.register_buffer("alpha_hats", torch.tensor(np.cumprod(1 - betas), dtype=torch.float32))

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule
        """
        # TODO(student): Implement cosine annealing beta schedule
        s = 0.08
        T = 1
        t = np.linspace(0, 1, timesteps + 1)
        f = np.cos(((t/T + s) / (1 + s)) * np.pi / 2) ** 2

        alpha_bars = f / f[0]

        betas_t = 1 - alpha_bars[1:] / alpha_bars[:-1]
        return np.clip(betas_t, 1e-4, 0.999)
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling
        """ 
        dt = 1 / self.flow_steps
        batch_size = observations.shape[0]
        x_t = noise
        # TODO(student): Implement DDPM sampling
        for i in reversed(range(self.flow_steps)):
            t = torch.full(
                (batch_size, 1),
                i * dt,
                device=observations.device,
                dtype=observations.dtype,
            )
            e_pred = self.actor(observations, x_t, t)
            alpha_t = self.alphas[i]
            alpha_hat_t = self.alpha_hats[i]
            beta_t = self.betas[i]

            mu = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_hat_t)) * e_pred)
            if i > 0:
                alpha_hat_prev = self.alpha_hats[i - 1]
                posterior_var = (
                    beta_t
                    * (1.0 - alpha_hat_prev)
                    / (1.0 - alpha_hat_t)
                )

                z = torch.randn_like(x_t)
                x_t = mu + torch.sqrt(posterior_var) * z
            else:
                x_t = mu
            
        return x_t

    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get_action
        observation = ptu.from_numpy(np.asarray(observation))[None]

        with torch.no_grad():
            z = torch.randn((1, self.action_dim), device=observation.device)
            actions = self.ddpm_sampler(observation, z)

        return ptu.to_numpy(actions)[0]

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
        Update Critic
        """
        # TODO(student): Implement critic update
        with torch.no_grad():
            z = torch.randn_like(actions)
            next_actions = self.ddpm_sampler(next_observations, z)
            target_q = self.target_critic(next_observations, next_actions).min(dim=0).values
            y = rewards + self.discount * (1 - dones) * target_q

        q = self.critic(observations, actions)
        loss = ((y - q)**2).mean()
        # TODO(student): Update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
                "q_loss": loss,
                "q_mean": q.mean(),
                "q_max": q.max(),
                "q_min": q.min(),
            }
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor
        """
        
        # TODO(student): Implement actor update
        
        # TODO(student): Update actor
        batch_size = observations.shape[0]
        
        z = torch.randn_like(actions)        
        t_idx = torch.randint(
            0,
            self.flow_steps,
            (batch_size,),
            device=actions.device,
        )
        alpha_hats_t = self.alpha_hats[t_idx].unsqueeze(-1)
        actions_noisy = torch.sqrt(alpha_hats_t) * actions + torch.sqrt(1 - alpha_hats_t) * z 
        actions_noisy =actions_noisy.detach().requires_grad_(True)
        t = t_idx.float().unsqueeze(-1) / self.flow_steps
        e_pred = self.actor(observations, actions_noisy, t)
        q_value = self.critic(observations, actions_noisy).min(dim=0).values

        q_grad = torch.autograd.grad(q_value.sum(), actions_noisy)[0]
        q_grad = q_grad.detach()
        loss_qsm = (- e_pred - self.inv_temp * q_grad).pow(2).mean()
        loss_denoise = (z - e_pred).pow(2).mean()
        loss = loss_qsm + self.alpha * loss_denoise

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return {
            "actor_loss": loss,
            "loss_qsm": loss_qsm,
            "loss_denoise": loss_denoise,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
                