"""Replay buffers for the DCPG value phase and the practice-match meta-update."""

import numpy as np
import torch


class PPGReplayBuffer:
    """Ring buffer of (obs, value target) pairs for the DCPG value phase."""

    def __init__(self, capacity, obs_shape, batch_size=32, track_mode_flags=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.track_mode_flags = track_mode_flags
        self.observations = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self.returns = torch.empty((capacity,), dtype=torch.float32)
        if track_mode_flags:
            self.mode_flags = torch.empty((capacity,), dtype=torch.float32)
        self.size = 0
        self._write_index = 0
        self.rng = np.random.default_rng(42)

    def push(self, obs: torch.Tensor, returns: torch.Tensor, mode_flags=None):
        obs = obs.detach().cpu().to(torch.uint8)
        returns = returns.detach().cpu()
        indices = np.array(range(self._write_index, self._write_index + obs.shape[0]))
        indices = indices % self.capacity
        self.observations[indices] = obs
        self.returns[indices] = returns
        if self.track_mode_flags and mode_flags is not None:
            self.mode_flags[indices] = mode_flags.detach().cpu()
        self._write_index = (indices[-1] + 1) % self.capacity
        self.size = min(self.size + obs.shape[0], self.capacity)

    def sample(self):
        indices = self.rng.choice(range(self.size), size=self.batch_size, replace=True)
        result = (
            self.observations[indices].float(),
            self.returns[indices],
        )
        if self.track_mode_flags:
            result = result + (self.mode_flags[indices],)
        return result


class MatchReplayBuffer:
    """Ring buffer of raw match-mode transitions for the practice-match meta-update."""

    def __init__(self, capacity, obs_shape, action_shape):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32)
        self.old_logprobs = torch.zeros(capacity, dtype=torch.float32)
        self.rewards_ex = torch.zeros(capacity, dtype=torch.float32)
        self.terminations = torch.zeros(capacity, dtype=torch.float32)

    def push(self, obs, next_obs, actions, old_logprobs, rewards_ex, terminations):
        n = obs.shape[0]
        if n == 0:
            return
        indices = (torch.arange(n) + self.pos) % self.capacity
        self.obs[indices] = obs.detach().cpu()
        self.next_obs[indices] = next_obs.detach().cpu()
        self.actions[indices] = actions.detach().cpu()
        self.old_logprobs[indices] = old_logprobs.detach().cpu()
        self.rewards_ex[indices] = rewards_ex.detach().cpu()
        self.terminations[indices] = terminations.detach().cpu()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        batch_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.obs[indices],
            self.next_obs[indices],
            self.actions[indices],
            self.old_logprobs[indices],
            self.rewards_ex[indices],
            self.terminations[indices],
        )

    def __len__(self):
        return self.size
