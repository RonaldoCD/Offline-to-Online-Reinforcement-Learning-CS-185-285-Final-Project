from infrastructure.utils import *


class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.masks = None

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observations": self.observations[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_observations": self.next_observations[idxs],
            "dones": self.dones[idxs],
            "masks": self.masks[idxs],
        }

    def sample_first(self, num_samples):
        num_samples = min(num_samples, self.size)
        idxs = np.arange(num_samples)

        return {
            "observations": self.observations[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_observations": self.next_observations[idxs],
            "dones": self.dones[idxs],
            "masks": self.masks[idxs],
        }

    
    def sample_sequence(self, batch_size, sequence_length, discount):        
        idxs = np.random.randint(self.size - sequence_length + 1, size=(batch_size,)) % self.max_size
        # print(f"Online samples (>500000): {np.sum(idxs >= 500000)} / {batch_size}")

        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        all_idxs = all_idxs.flatten()

        batch_actions = self.actions[all_idxs].reshape(batch_size, sequence_length, *self.actions.shape[1:])
        batch_rewards = self.rewards[all_idxs].reshape(batch_size, sequence_length, *self.rewards.shape[1:])
        batch_dones = self.dones[all_idxs].reshape(batch_size, sequence_length, *self.dones.shape[1:])
        batch_masks = self.masks[all_idxs].reshape(batch_size, sequence_length, *self.masks.shape[1:])
        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        dones = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        
        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        dones[:, 0] = batch_dones[:, 0].squeeze()  
        masks[:, 0] = batch_masks[:, 0].squeeze()      
        discount_powers = discount ** np.arange(sequence_length)

        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            dones[:, i] = np.maximum(dones[:, i-1], batch_dones[:, i].squeeze())
            valid[:, i] = 1.0 - dones[:, i-1]

        return {
            "observations": self.observations[idxs],
            "actions": batch_actions,
            "rewards": rewards,
            "next_observations": self.next_observations[idxs + sequence_length - 1],
            "dones": dones,
            "valids": valid,
            "masks": masks,
        }
            
    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: np.ndarray,
        done: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(action, int):
            action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.empty(
                (self.max_size, *observation.shape), dtype=observation.dtype
            )
            self.actions = np.empty((self.max_size, *action.shape), dtype=action.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.empty(
                (self.max_size, *next_observation.shape), dtype=next_observation.dtype
            )
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.masks = np.empty((self.max_size, *mask.shape), dtype=mask.dtype)

        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        assert reward.shape == ()
        assert next_observation.shape == self.next_observations.shape[1:]
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.masks[self.size % self.max_size] = mask

        self.size += 1
