"""State representation: 11 features describing the game for the agent.

Feature order:
    0-2   danger straight, right, left (relative to the current heading)
    3-6   heading one-hot: up, right, down, left
    7-10  food relative to the head: up, right, down, left
"""

from __future__ import annotations

import numpy as np

from snake_ai.game import Direction, Point, SnakeGame

STATE_SIZE = 11

_CLOCKWISE = list(Direction)


def get_state(game: SnakeGame) -> np.ndarray:
    """Encode the game as a float32 vector of length STATE_SIZE."""
    head = game.head
    index = _CLOCKWISE.index(game.direction)
    straight = _CLOCKWISE[index]
    right = _CLOCKWISE[(index + 1) % 4]
    left = _CLOCKWISE[(index - 1) % 4]

    def danger(direction: Direction) -> bool:
        step = direction.value
        return game.is_collision(Point(head.x + step.x, head.y + step.y))

    food = game.food
    if food is None:
        food_flags = [False, False, False, False]
    else:
        food_flags = [food.y < head.y, food.x > head.x, food.y > head.y, food.x < head.x]

    heading = [game.direction is d for d in _CLOCKWISE]
    features = [danger(straight), danger(right), danger(left), *heading, *food_flags]
    return np.array(features, dtype=np.float32)
