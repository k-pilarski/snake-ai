"""Play Snake with the keyboard: python -m snake_ai.play_manual

Controls: arrow keys to steer, R to restart after game over, Esc to quit.
"""

from __future__ import annotations

import pygame

from snake_ai.game import Direction, SnakeGame
from snake_ai.render import Renderer

KEY_TO_DIRECTION = {
    pygame.K_UP: Direction.UP,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
}


def main() -> None:
    game = SnakeGame()
    renderer = Renderer(game.width, game.height)
    direction = game.direction
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game.game_over:
                    game.reset()
                    direction = game.direction
                elif event.key in KEY_TO_DIRECTION:
                    direction = KEY_TO_DIRECTION[event.key]

        if not game.game_over:
            game.move(direction)

        message = "Game over - R to restart, Esc to quit" if game.game_over else None
        renderer.draw(game, message)
        renderer.tick()

    renderer.close()


if __name__ == "__main__":
    main()
