import pytest

from snake_ai.env import (
    NUM_ACTIONS,
    REWARD_DEATH,
    REWARD_FOOD,
    STRAIGHT,
    TIMEOUT_FACTOR,
    TURN_LEFT,
    TURN_RIGHT,
    SnakeEnv,
)
from snake_ai.game import Direction, Point


def make_env() -> SnakeEnv:
    env = SnakeEnv(seed=0)
    env.reset()
    env.game.food = Point(0, 0)  # keep food out of the way
    return env


def test_num_actions():
    assert NUM_ACTIONS == 3


def test_reset_restores_episode():
    env = make_env()
    env.step(STRAIGHT)
    env.reset()
    assert env.game.score == 0
    assert env.steps_since_food == 0
    assert not env.game.game_over


def test_straight_keeps_direction():
    env = make_env()
    env.step(STRAIGHT)
    assert env.game.direction is Direction.RIGHT


def test_turn_right_from_right_goes_down():
    env = make_env()
    env.step(TURN_RIGHT)
    assert env.game.direction is Direction.DOWN


def test_turn_left_from_right_goes_up():
    env = make_env()
    env.step(TURN_LEFT)
    assert env.game.direction is Direction.UP


def test_reward_for_food():
    env = make_env()
    env.game.food = Point(11, 10)
    _, reward, done, info = env.step(STRAIGHT)
    assert reward == REWARD_FOOD
    assert not done
    assert info["score"] == 1
    assert info["reason"] == "food"


def test_reward_for_death():
    env = make_env()
    for _ in range(9):
        env.step(STRAIGHT)
    _, reward, done, info = env.step(STRAIGHT)
    assert reward == REWARD_DEATH
    assert done
    assert info["reason"] == "wall_or_self"


def test_zero_reward_for_normal_step():
    env = make_env()
    _, reward, done, info = env.step(STRAIGHT)
    assert reward == 0.0
    assert not done
    assert info["reason"] is None


def test_timeout_ends_episode():
    env = make_env()
    env.steps_since_food = TIMEOUT_FACTOR * len(env.game.snake) - 1
    _, reward, done, info = env.step(STRAIGHT)
    assert reward == REWARD_DEATH
    assert done
    assert info["reason"] == "timeout"


def test_eating_resets_timeout_counter():
    env = make_env()
    env.steps_since_food = 50
    env.game.food = Point(11, 10)
    env.step(STRAIGHT)
    assert env.steps_since_food == 0


def test_invalid_action_raises():
    env = make_env()
    with pytest.raises(ValueError):
        env.step(3)
