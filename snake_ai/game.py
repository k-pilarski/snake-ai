"""Core Snake game logic.

This module is headless: it has no graphics and no dependency on pygame,
so it can be used for fast training and for rendering alike.
"""

from __future__ import annotations

import random
from collections import deque
from enum import Enum
from typing import NamedTuple


class Point(NamedTuple):
    """A cell on the grid. The origin is the top-left corner, y grows downwards."""

    x: int
    y: int


class Direction(Enum):
    """Movement directions, listed clockwise."""

    UP = Point(0, -1)
    RIGHT = Point(1, 0)
    DOWN = Point(0, 1)
    LEFT = Point(-1, 0)

    def opposite(self) -> Direction:
        return Direction(Point(-self.value.x, -self.value.y))


MIN_GRID_SIZE = 5
INITIAL_LENGTH = 3


class SnakeGame:
    """Snake on a rectangular grid with solid walls."""

    def __init__(self, width: int = 20, height: int = 20, seed: int | None = None) -> None:
        if width < MIN_GRID_SIZE or height < MIN_GRID_SIZE:
            raise ValueError(f"Grid must be at least {MIN_GRID_SIZE}x{MIN_GRID_SIZE}.")
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.snake: deque[Point] = deque()
        self.direction = Direction.RIGHT
        self.food: Point | None = None
        self.score = 0
        self.game_over = False
        self.reset()

    @property
    def head(self) -> Point:
        return self.snake[0]

    def reset(self) -> None:
        """Start a new game: snake in the middle heading right, one piece of food."""
        head = Point(self.width // 2, self.height // 2)
        self.direction = Direction.RIGHT
        self.snake = deque(Point(head.x - i, head.y) for i in range(INITIAL_LENGTH))
        self.score = 0
        self.game_over = False
        self.food = self._spawn_food()

    def is_wall(self, point: Point) -> bool:
        return not (0 <= point.x < self.width and 0 <= point.y < self.height)

    def is_collision(self, point: Point) -> bool:
        """Return True if `point` is outside the grid or occupied by the snake."""
        return self.is_wall(point) or point in self.snake

    def move(self, direction: Direction) -> bool:
        """Advance the snake by one cell.

        A request to turn 180 degrees is ignored and the snake keeps its direction.
        Returns True if the snake ate food during this move. After a fatal move
        `game_over` is set to True.
        """
        if self.game_over:
            raise RuntimeError("Game is over. Call reset() to start a new game.")

        if direction is not self.direction.opposite():
            self.direction = direction

        step = self.direction.value
        new_head = Point(self.head.x + step.x, self.head.y + step.y)
        ate_food = new_head == self.food

        # Without growth the tail leaves its cell, so the head may enter it.
        occupied = set(self.snake) if ate_food else set(list(self.snake)[:-1])
        if self.is_wall(new_head) or new_head in occupied:
            self.game_over = True
            return False

        self.snake.appendleft(new_head)
        if ate_food:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:  # the board is full: the game is won
                self.game_over = True
        else:
            self.snake.pop()
        return ate_food

    def _spawn_food(self) -> Point | None:
        """Place food on a random free cell, or return None if there is none."""
        occupied = set(self.snake)
        free = [
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if Point(x, y) not in occupied
        ]
        if not free:
            return None
        return self._rng.choice(free)
