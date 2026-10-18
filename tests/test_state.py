import numpy as np

from snake_ai.env import STRAIGHT, SnakeEnv
from snake_ai.game import Direction, Point, SnakeGame
from snake_ai.state import STATE_SIZE, get_state


def make_game(snake, direction, food=None):
    game = SnakeGame(seed=0)
    game.snake.clear()
    game.snake.extend(snake)
    game.direction = direction
    game.food = food if food is not None else Point(0, 0)
    return game


def test_state_shape_and_dtype():
    state = get_state(SnakeGame(seed=0))
    assert STATE_SIZE == 11
    assert state.shape == (11,)
    assert state.dtype == np.float32


def test_initial_state_values():
    game = SnakeGame(seed=0)
    game.food = Point(10, 5)
    expected = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    np.testing.assert_array_equal(get_state(game), expected)


def test_danger_straight_at_wall():
    game = make_game([Point(19, 10), Point(18, 10), Point(17, 10)], Direction.RIGHT)
    np.testing.assert_array_equal(get_state(game)[:3], [1, 0, 0])


def test_danger_left_at_wall():
    game = make_game([Point(10, 0), Point(9, 0), Point(8, 0)], Direction.RIGHT)
    np.testing.assert_array_equal(get_state(game)[:3], [0, 0, 1])


def test_danger_from_body():
    snake = [Point(5, 5), Point(5, 6), Point(4, 6), Point(4, 5), Point(4, 4)]
    game = make_game(snake, Direction.UP)
    np.testing.assert_array_equal(get_state(game)[:3], [0, 0, 1])


def test_danger_is_relative_to_heading():
    game = make_game([Point(10, 19), Point(10, 18), Point(10, 17)], Direction.DOWN)
    state = get_state(game)
    np.testing.assert_array_equal(state[:3], [1, 0, 0])
    np.testing.assert_array_equal(state[3:7], [0, 0, 1, 0])


def test_food_direction_diagonal():
    game = SnakeGame(seed=0)
    game.food = Point(15, 5)
    np.testing.assert_array_equal(get_state(game)[7:], [1, 1, 0, 0])


def test_food_on_same_row_has_no_vertical_flag():
    game = SnakeGame(seed=0)
    game.food = Point(15, 10)
    np.testing.assert_array_equal(get_state(game)[7:], [0, 1, 0, 0])


def test_no_food_gives_zero_food_features():
    game = SnakeGame(seed=0)
    game.food = None
    np.testing.assert_array_equal(get_state(game)[7:], [0, 0, 0, 0])


def test_env_reset_returns_state():
    state = SnakeEnv(seed=0).reset()
    assert state.shape == (11,)


def test_env_step_returns_state():
    env = SnakeEnv(seed=0)
    env.reset()
    state, _, _, _ = env.step(STRAIGHT)
    assert state.shape == (11,)
    assert state.dtype == np.float32
