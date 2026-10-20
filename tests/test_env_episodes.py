import random

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


def test_win_on_full_board_has_no_penalty():
    env = SnakeEnv(width=5, height=5, seed=0)
    cells = [
        Point(x, y)
        for y in range(5)
        for x in (range(5) if y % 2 == 0 else reversed(range(5)))
    ]
    env.game.snake.clear()
    env.game.snake.extend(cells[1:])  # head at (1, 0), the only free cell is (0, 0)
    env.game.direction = Direction.LEFT
    env.game.food = Point(0, 0)

    _, reward, done, info = env.step(STRAIGHT)

    assert reward == REWARD_FOOD
    assert done
    assert info["reason"] == "win"
    assert info["score"] == 1
    assert env.game.food is None


def test_timeout_threshold_scales_with_snake_length():
    env = make_env()
    env.game.snake.append(Point(7, 10))  # length 4
    env.steps_since_food = TIMEOUT_FACTOR * 4 - 2

    _, reward, done, _ = env.step(STRAIGHT)
    assert reward == 0.0
    assert not done

    _, reward, done, info = env.step(STRAIGHT)
    assert reward == REWARD_DEATH
    assert done
    assert info["reason"] == "timeout"


def test_self_collision_gives_death_reward():
    env = make_env()
    env.game.snake.clear()
    env.game.snake.extend([Point(5, 5), Point(5, 6), Point(4, 6), Point(4, 5), Point(4, 4)])
    env.game.direction = Direction.UP

    _, reward, done, info = env.step(TURN_LEFT)

    assert reward == REWARD_DEATH
    assert done
    assert info["reason"] == "wall_or_self"


def run_episode(seed: int) -> list:
    env = SnakeEnv(seed=seed)
    env.reset()
    rng = random.Random(123)
    trace = []
    for _ in range(500):
        state, reward, done, info = env.step(rng.randrange(NUM_ACTIONS))
        trace.append((state.tolist(), reward, done, info["score"]))
        if done:
            break
    return trace


def test_same_seed_and_actions_give_same_episode():
    assert run_episode(7) == run_episode(7)


def test_random_episodes_keep_invariants():
    for seed in range(5):
        env = SnakeEnv(width=8, height=8, seed=seed)
        env.reset()
        rng = random.Random(seed)
        done = False
        for _ in range(20000):
            state, reward, done, _ = env.step(rng.randrange(NUM_ACTIONS))
            game = env.game
            assert state.shape == (11,)
            assert reward in (REWARD_FOOD, REWARD_DEATH, 0.0)
            assert len(set(game.snake)) == len(game.snake)
            assert all(not game.is_wall(p) for p in game.snake)
            assert game.food not in game.snake
            if done:
                break
        assert done


def test_state_reflects_heading_after_turn():
    env = make_env()
    state, _, _, _ = env.step(TURN_RIGHT)
    assert state[3:7].tolist() == [0, 0, 1, 0]  # heading down
