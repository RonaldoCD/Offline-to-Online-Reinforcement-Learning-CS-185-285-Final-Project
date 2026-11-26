from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class QAMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
 
        make_actor_flow,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float, 
        flow_steps: int,
        horizon: int,
        num_samples: int = 1,
        q_agg: str = 'mean',
        num_critics: int = 2,
        eval_flow_type: str = "qam",  # set to "bc" to evaluate the BC flow policy alone
        max_grad_norm: float = 1.0,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.chunk_size = self.action_dim * self.horizon
        
        # TODO(student): Create actor
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        self.actor_qam = make_actor_flow(observation_shape, self.chunk_size)
        self.actor_flow_bc = make_actor_flow(observation_shape, self.chunk_size)
        self.critic = make_critic(observation_shape, self.chunk_size, num_critics=num_critics)
        self.target_critic = make_critic(observation_shape, self.chunk_size, num_critics=num_critics)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # TODO(student): Create optimizers for all the above models
        self.actor_qam_optimizer = make_actor_optimizer(self.actor_qam.parameters())
        self.actor_flow_bc_optimizer = make_actor_optimizer(self.actor_flow_bc.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps
        self.h = 1 / self.flow_steps
        self.q_agg = q_agg
        self.num_samples = num_samples
        self.eval_flow_type = eval_flow_type
        self.max_grad_norm = max_grad_norm

    def sample_actions(self, observations: torch.Tensor):
        for n in range(self.num_samples):
            z = torch.randn((observations.shape[0], self.chunk_size), device=observations.device, dtype=observations.dtype)
            actions = self.get_flow_action(observations, z, flow_type=self.eval_flow_type)
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

    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        with torch.no_grad():
            action = self.sample_actions(observation)
        
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor, flow_type: str = "qam") -> torch.Tensor:
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
            if flow_type == "qam":
                velocity = self.actor_qam(observation, action, t)
            else:
                velocity = self.actor_flow_bc(observation, action, t)
            action = action + dt * velocity
            if not torch.isfinite(action).all():
                raise RuntimeError(
                    f"Invalid generated action at flow step {i}"
                )
        action = torch.clamp(action, -1, 1)
        return action

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
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, -1)

        z = torch.randn_like(actions)
        t = torch.rand((batch_size, 1),
                        device=actions.device,
                        dtype=actions.dtype)
        
        actions_hat = (1 - t) * z + t * actions
        velocities_pred = self.actor_flow_bc(observations, actions_hat, t)
        velocities_target = actions - z
        flow_error = (velocities_pred - velocities_target)**2
        # Whole-chunk mask, as in QAM
        loss = torch.mean(flow_error * valids[:, -1].unsqueeze(-1))

        self.actor_flow_bc_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_flow_bc.parameters(), self.max_grad_norm)
        self.actor_flow_bc_optimizer.step()

        return {
            "loss": loss,
        }

    def get_sde(self,
                observations: torch.Tensor, 
                a0: torch.Tensor):
        
        batch_size = observations.shape[0]
        a_ts = [a0]
        ts = []
        
        for i in range(self.flow_steps):
            t = torch.ones((batch_size, 1), 
                            device=observations.device,
                            dtype=observations.dtype) * i * self.h
            t_shifted = t + self.h
            a_prev = a_ts[-1]
            # Memoryless noise with the +h offsets used by QAM (keeps sigma > 0 at every step)
            sigma_t = torch.sqrt(2 * self.h * (1 - t + self.h) / t_shifted)
            z_t = torch.randn_like(a0)            
            if i != self.flow_steps - 1:
                velocity = self.actor_qam(observations, a_prev, t)
                a_next = a_prev + self.h * (2 * velocity - a_prev / t_shifted) + sigma_t * z_t
            else:
                # Last step: deterministic ODE step with the BC actor
                a_next = a_prev + self.h * self.actor_flow_bc(observations, a_prev, t)
            
            if not torch.isfinite(a_next).all():
                raise RuntimeError(
                    f"Invalid SDE state at step {i}: "
                    f"min={a_next.nan_to_num().min().item()}, "
                    f"max={a_next.nan_to_num().max().item()}"
                )
            a_ts.append(a_next)
            ts.append(t)

        return torch.stack(a_ts[:-1], dim=0), torch.stack(ts, dim=0), a_ts[-1]

    def get_adjoints(self, observations, a_ts, g1):
        batch_size = observations.shape[0]
        g_ts = [g1]
        for i in reversed(range(self.flow_steps)):
            t = torch.ones((batch_size, 1), 
                            device=observations.device,
                            dtype=observations.dtype) * i * self.h
            t_shifted = t + self.h
            a_t = a_ts[i].detach().requires_grad_(True)
            # BC velocity evaluated at t + h, as in QAM
            by = 2 * self.actor_flow_bc(observations, a_t, t_shifted) - a_t / t_shifted
            vjp = torch.autograd.grad(
                outputs=by,
                inputs=a_t,
                grad_outputs=g_ts[-1],
                create_graph=False,
            )[0]
            gt = g_ts[-1] + self.h * vjp
            if not torch.isfinite(gt).all():
                raise RuntimeError(
                    f"Invalid adjoint at step {i}"
                )
            g_ts.append(gt)   
        return torch.stack(list(reversed(g_ts[1:])), dim=0)

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
            next_actions = torch.clamp(self.get_flow_action(next_observations, z, flow_type="qam"), -1, 1) ## ToFix: Check next_actions shape
            # print(f"next_actions shape: {next_actions.shape}, next_observations shape: {next_observations.shape}")
            if self.q_agg == 'mean':
                q_next = self.target_critic(next_observations, next_actions).mean(dim=0)                
            else:
                q_next = self.target_critic(next_observations, next_actions)
                q_next = q_next.min(dim=0).values

            y = rewards[:, -1] + (self.discount**self.horizon) * masks[:,-1] * q_next 

        q = self.critic(observations, actions)
        # print(f"q shape: {q.shape}, y shape: {y.shape}, valids shape: {valids.shape}, actions shape: {actions.shape}" )
        critic_w = valids[:, -1]
        loss = (((q - y.unsqueeze(0)) ** 2) * critic_w).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
     
    @torch.compile()
    def update_qam_actor(self, observations, actions):
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, -1)

        a0 = torch.randn_like(actions)

        # Generate SDE trajectory
        with torch.no_grad():
            a_ts, ts, a1 = self.get_sde(
                observations, a0
            )

        with torch.enable_grad():
            a1 = a1.detach().requires_grad_(True)

            q = self.target_critic(observations, torch.clamp(a1, -1, 1)).mean(dim=0)

            g1 = -self.inv_temp * torch.autograd.grad(
                outputs=q.sum(),
                inputs=a1,
            )[0]

        g_ts = self.get_adjoints(
            observations, a_ts, g1.detach()
        )

        sigmas = torch.sqrt(2 * (1 - ts + self.h) / (ts + self.h))

        observations_ts = (
            observations.unsqueeze(0)
            .expand(self.flow_steps, -1, -1)
            .reshape(self.flow_steps * batch_size, -1)
        )

        a_ts_flat = a_ts.reshape(
            self.flow_steps * batch_size, -1
        )

        ts_flat = ts.reshape(
            self.flow_steps * batch_size, 1
        )

        vel_qam = self.actor_qam(observations_ts, a_ts_flat, ts_flat).reshape(self.flow_steps, batch_size, -1)

        with torch.no_grad():
            vel_bc = self.actor_flow_bc(observations_ts, a_ts_flat, ts_flat).reshape(self.flow_steps, batch_size, -1)

        vel_diff = vel_qam - vel_bc
        residual = 2 * vel_diff / sigmas + sigmas * g_ts
        loss = residual.square().sum(dim=-1).sum(dim=0).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Invalid QAM loss: {loss.item()}"
            )

        self.actor_qam_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_qam.parameters(), self.max_grad_norm)
        self.actor_qam_optimizer.step()

        return {
            "qam_loss": loss,
            # |dQ/da| at the generated action. If ~0, the critic is flat: no learning signal.
            "grad_q_norm": (g1 / self.inv_temp).norm(dim=-1).mean(),
            "adjoint_norm_t0": g_ts[0].norm(dim=-1).mean(),
            "adjoint_norm_t1": g1.norm(dim=-1).mean(),
            "vel_diff_norm": vel_diff.norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def compute_diagnostics(self, observations, actions):
        """
        More expensive diagnostics: Q of dataset / BC / QAM actions, distance between
        the QAM and BC policies, and action saturation. Run every `diag_interval` steps.
        """
        batch_size = actions.shape[0]
        actions = actions.reshape(batch_size, -1)
        z = torch.randn_like(actions)
        a_bc = self.get_flow_action(observations, z, flow_type="bc")
        a_qam_raw = self._flow_raw(observations, z)
        a_qam = torch.clamp(a_qam_raw, -1, 1)

        q_data = self.critic(observations, actions).mean(dim=0)
        q_bc = self.critic(observations, a_bc).mean(dim=0)
        q_qam = self.critic(observations, a_qam).mean(dim=0)

        return {
            "q_data": q_data.mean(),
            "q_bc": q_bc.mean(),
            "q_qam": q_qam.mean(),
            # Should be > 0 if the QAM actor actually improves over BC under the critic.
            "q_qam_minus_bc": (q_qam - q_bc).mean(),
            "qam_bc_action_mse": (a_qam - a_bc).square().mean(),
            "qam_data_action_mse": (a_qam - actions).square().mean(),
            "action_saturation": (a_qam_raw.abs() > 0.99).float().mean(),
            "action_abs_max": a_qam_raw.abs().max(),
        }

    def _flow_raw(self, observations, noise):
        """QAM ODE sample without clamping (for saturation diagnostics)."""
        action = noise
        for i in range(self.flow_steps):
            t = torch.full((noise.shape[0], 1), i * self.h,
                           device=observations.device, dtype=observations.dtype)
            action = action + self.h * self.actor_qam(observations, action, t)
        return action
    
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
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones, valids, masks)
        metrics_bc_actor = self.update_bc_actor(observations, actions, valids)
        metric_qam_actor = self.update_qam_actor(observations, actions)

        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"qam_actor/{k}": v.item() for k, v in metric_qam_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.target_update_rate * param.data + (1 - self.target_update_rate) * target_param.data)
                