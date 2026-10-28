"""DQN agent: epsilon-greedy policy, replay memory and the learning step."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from snake_ai.dqn_model import DQNModel
from snake_ai.env import NUM_ACTIONS
from snake_ai.replay_buffer import ReplayBuffer


class DQNAgent:
    """Learns Q-values with a neural network trained on batches from a replay buffer.

    The target of every update comes from a separate target network that is
    synchronized with the main model every `target_update_every` learning steps.
    If `seed` is given, it also calls torch.manual_seed, which seeds PyTorch
    globally because the initial weights are drawn from the global generator.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_capacity: int = 100_000,
        target_update_every: int = 100,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.learn_steps = 0

        self.model = DQNModel()
        self.target_model = DQNModel()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        buffer_seed = None if seed is None else seed + 1
        self.memory = ReplayBuffer(memory_capacity, seed=buffer_seed)
        self._rng = random.Random(seed)

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy: explore with probability epsilon, otherwise take the best action."""
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(NUM_ACTIONS)
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
            q_values = self.model(tensor)
        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def compute_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Bellman targets: r + gamma * max Q_target(s'), or just r for terminal transitions."""
        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1).values
            return rewards + self.gamma * next_q * (1.0 - dones)

    def train_step(self) -> float | None:
        """Run one learning step on a random batch and return the loss.

        Returns None if the memory holds fewer transitions than the batch size.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.from_numpy(states)
        actions_t = torch.from_numpy(actions)
        rewards_t = torch.from_numpy(rewards)
        next_states_t = torch.from_numpy(next_states)
        dones_t = torch.from_numpy(dones)

        q_values = self.model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        targets = self.compute_targets(rewards_t, next_states_t, dones_t)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Lower epsilon after an episode, never below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
