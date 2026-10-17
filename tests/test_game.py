import pytest

from snake_ai.game import Direction, Point, SnakeGame


def test_initial_state():
    game = SnakeGame(width=20, height=20, seed=0)
    assert len(game.snake) == 3
    assert game.head == Point(10, 10)
    assert game.direction is Direction.RIGHT
    assert game.score == 0
    assert not game.game_over
    assert game.food is not None
    assert game.food not in game.snake


def test_grid_size_is_configurable():
    game = SnakeGame(width=8, height=6)
    assert (game.width, game.height) == (8, 6)
    assert all(0 <= p.x < 8 and 0 <= p.y < 6 for p in game.snake)


def test_grid_too_small_is_rejected():
    with pytest.raises(ValueError):
        SnakeGame(width=4, height=10)


def test_snake_moves_without_growing():
    game = SnakeGame(seed=0)
    game.food = Point(0, 0)
    ate = game.move(Direction.RIGHT)
    assert not ate
    assert game.head == Point(11, 10)
    assert len(game.snake) == 3


def test_snake_grows_and_scores_after_eating():
    game = SnakeGame(seed=0)
    game.food = Point(11, 10)
    ate = game.move(Direction.RIGHT)
    assert ate
    assert len(game.snake) == 4
    assert game.score == 1
    assert game.food is not None
    assert game.food not in game.snake


def test_wall_collision_ends_game():
    game = SnakeGame(seed=0)
    game.food = Point(0, 0)
    for _ in range(10):
        game.move(Direction.UP)
    assert game.head == Point(10, 0)
    assert not game.game_over
    game.move(Direction.UP)
    assert game.game_over


def test_self_collision_ends_game():
    game = SnakeGame(seed=0)
    game.food = Point(0, 0)
    game.snake.clear()
    game.snake.extend([Point(5, 5), Point(5, 6), Point(4, 6), Point(4, 5), Point(4, 4)])
    game.direction = Direction.UP
    game.move(Direction.LEFT)
    assert game.game_over


def test_head_may_enter_cell_left_by_tail():
    game = SnakeGame(seed=0)
    game.food = Point(0, 0)
    game.snake.clear()
    game.snake.extend([Point(5, 5), Point(5, 6), Point(4, 6), Point(4, 5)])
    game.direction = Direction.UP
    game.move(Direction.LEFT)
    assert not game.game_over
    assert game.head == Point(4, 5)


def test_reverse_direction_is_ignored():
    game = SnakeGame(seed=0)
    game.food = Point(0, 0)
    game.move(Direction.LEFT)
    assert game.direction is Direction.RIGHT
    assert game.head == Point(11, 10)
    assert not game.game_over


def test_move_after_game_over_raises():
    game = SnakeGame(seed=0)
    game.game_over = True
    with pytest.raises(RuntimeError):
        game.move(Direction.UP)


def test_food_never_spawns_on_snake():
    for seed in range(50):
        game = SnakeGame(width=6, height=6, seed=seed)
        assert game.food not in game.snake


def test_food_spawns_on_the_only_free_cell():
    game = SnakeGame(width=5, height=5, seed=0)
    cells = [
        Point(x, y)
        for y in range(5)
        for x in (range(5) if y % 2 == 0 else reversed(range(5)))
    ]
    game.snake.clear()
    game.snake.extend(cells[:-1])
    assert game._spawn_food() == cells[-1]


def test_same_seed_gives_same_food():
    assert SnakeGame(seed=42).food == SnakeGame(seed=42).food


def test_reset_restores_initial_state():
    game = SnakeGame(seed=0)
    game.food = Point(11, 10)
    game.move(Direction.RIGHT)
    game.reset()
    assert game.score == 0
    assert len(game.snake) == 3
    assert not game.game_over
