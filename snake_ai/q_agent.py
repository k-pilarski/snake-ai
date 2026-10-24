"""Tabular Q-learning agent with an epsilon-greedy policy."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from snake_ai.env import NUM_ACTIONS

_HYPERPARAMETERS = ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay")


class QAgent:
    """Stores a value for every (state, action) pair and learns with Q-learning."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self._rng = random.Random(seed)

    @staticmethod
    def _key(state: np.ndarray) -> tuple[int, ...]:
        return tuple(int(v) for v in state)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return the Q-values of a state, creating a zero entry on first visit."""
        key = self._key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self.q_table[key]

    def best_action(self, state: np.ndarray) -> int:
        """Return the action with the highest Q-value, breaking ties at random."""
        values = self.q_values(state)
        best = max(values)
        candidates = [a for a in range(NUM_ACTIONS) if values[a] == best]
        return self._rng.choice(candidates)

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy: explore with probability epsilon, otherwise exploit."""
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(NUM_ACTIONS)
        return self.best_action(state)

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Apply one Q-learning update for the observed transition."""
        values = self.q_values(state)
        future = 0.0 if done else float(np.max(self.q_values(next_state)))
        target = reward + self.gamma * future
        values[action] += self.alpha * (target - values[action])

    def decay_epsilon(self) -> None:
        """Lower epsilon after an episode, never below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Write the Q-table and hyperparameters to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hyperparameters": {name: getattr(self, name) for name in _HYPERPARAMETERS},
            "q_table": {
                ",".join(str(v) for v in key): values.tolist()
                for key, values in self.q_table.items()
            },
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> QAgent:
        """Read an agent saved with save(). The random generator is seeded with `seed`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(seed=seed, **data["hyperparameters"])
        for text, values in data["q_table"].items():
            key = tuple(int(v) for v in text.split(","))
            agent.q_table[key] = np.array(values, dtype=np.float64)
        return agent
