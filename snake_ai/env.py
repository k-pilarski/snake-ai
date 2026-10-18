"""Gym-style environment wrapping the Snake game.

Actions are relative to the current heading: 0 = straight, 1 = turn right
(clockwise), 2 = turn left (counter-clockwise).
"""

from __future__ import annotations

from typing import Any

from snake_ai.game import Direction, SnakeGame

STRAIGHT = 0
TURN_RIGHT = 1
TURN_LEFT = 2
NUM_ACTIONS = 3

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
TIMEOUT_FACTOR = 100

_CLOCKWISE = list(Direction)


class SnakeEnv:
    """Environment with `reset()` and `step(action)` in the style of Gymnasium."""

    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None) -> None:
        self.game = SnakeGame(width=width, height=height, seed=seed)
        self.steps_since_food = 0

    def reset(self) -> Any:
        """Start a new episode and return the initial observation."""
        self.game.reset()
        self.steps_since_food = 0
        return self._observe()

    def step(self, action: int) -> tuple[Any, float, bool, dict[str, Any]]:
        """Apply an action and return (observation, reward, done, info)."""
        if action not in (STRAIGHT, TURN_RIGHT, TURN_LEFT):
            raise ValueError(f"Invalid action: {action}. Expected 0, 1 or 2.")

        direction = self._new_direction(action)
        ate_food = self.game.move(direction)
        self.steps_since_food = 0 if ate_food else self.steps_since_food + 1

        reward = 0.0
        reason: str | None = None
        done = False

        if ate_food:
            reward = REWARD_FOOD
            reason = "food"
            if self.game.game_over:  # the board is full
                done = True
                reason = "win"
        elif self.game.game_over:
            reward = REWARD_DEATH
            done = True
            reason = "wall_or_self"
        elif self.steps_since_food >= TIMEOUT_FACTOR * len(self.game.snake):
            reward = REWARD_DEATH
            done = True
            reason = "timeout"

        info = {"score": self.game.score, "reason": reason}
        return self._observe(), reward, done, info

    def _new_direction(self, action: int) -> Direction:
        index = _CLOCKWISE.index(self.game.direction)
        if action == TURN_RIGHT:
            index = (index + 1) % 4
        elif action == TURN_LEFT:
            index = (index - 1) % 4
        return _CLOCKWISE[index]

    def _observe(self) -> Any:
        """Return the observation. The 11-feature state is added in a later issue."""
        return None
