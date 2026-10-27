"""Replay buffer: a fixed-size memory of past transitions for DQN training."""

from __future__ import annotations

import random

import numpy as np

Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Stores transitions and returns random batches.

    When the buffer is full, the oldest transition is overwritten.
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.capacity = capacity
        self._storage: list[Transition] = []
        self._next = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition, overwriting the oldest one if the buffer is full."""
        transition: Transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a random batch without repetitions.

        The result is (states, actions, rewards, next_states, dones).
        """
        if batch_size > len(self._storage):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of {len(self._storage)}."
            )
        indices = self._rng.sample(range(len(self._storage)), batch_size)
        batch = [self._storage[i] for i in indices]
        states = np.stack([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones
