"""Optional pygame renderer for the Snake game.

The game logic never imports this module, so training can run without a window.
"""

from __future__ import annotations

import pygame

from snake_ai.game import SnakeGame

BACKGROUND = (18, 18, 18)
GRID_LINE = (32, 32, 32)
SNAKE_HEAD = (80, 220, 120)
SNAKE_BODY = (40, 160, 80)
FOOD = (230, 70, 70)
TEXT = (240, 240, 240)


class Renderer:
    """Draws a SnakeGame in a pygame window."""

    def __init__(self, width: int, height: int, cell_size: int = 30, fps: int = 10) -> None:
        pygame.init()
        self.cell_size = cell_size
        self.fps = fps
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption("snake-ai")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

    def draw(self, game: SnakeGame, message: str | None = None) -> None:
        """Draw the current state of the game and show it on screen."""
        size = self.cell_size
        self.screen.fill(BACKGROUND)

        for x in range(game.width):
            for y in range(game.height):
                cell = pygame.Rect(x * size, y * size, size, size)
                pygame.draw.rect(self.screen, GRID_LINE, cell, width=1)

        if game.food is not None:
            food_rect = pygame.Rect(game.food.x * size, game.food.y * size, size, size)
            pygame.draw.rect(self.screen, FOOD, food_rect.inflate(-6, -6), border_radius=6)

        for index, point in enumerate(game.snake):
            color = SNAKE_HEAD if index == 0 else SNAKE_BODY
            rect = pygame.Rect(point.x * size, point.y * size, size, size)
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4), border_radius=4)

        label = self.font.render(f"Score: {game.score}", True, TEXT)
        self.screen.blit(label, (8, 6))
        if message:
            note = self.font.render(message, True, TEXT)
            self.screen.blit(note, (8, 6 + label.get_height()))

        pygame.display.flip()

    def tick(self) -> None:
        """Wait so that the game runs at the configured frames per second."""
        self.clock.tick(self.fps)

    def close(self) -> None:
        pygame.quit()
