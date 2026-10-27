import numpy as np
import pytest

from snake_ai.replay_buffer import ReplayBuffer


def push_numbered(buffer: ReplayBuffer, count: int) -> None:
    """Push transitions whose contents all encode their index i."""
    for i in range(count):
        buffer.push(
            np.full(11, i, dtype=np.float32),
            i % 3,
            float(i),
            np.full(11, i + 1, dtype=np.float32),
            i % 2 == 0,
        )


def test_length_grows_until_capacity_and_stays():
    buffer = ReplayBuffer(capacity=5)
    assert len(buffer) == 0
    push_numbered(buffer, 3)
    assert len(buffer) == 3
    push_numbered(buffer, 10)
    assert len(buffer) == 5


def test_oldest_transitions_are_dropped():
    buffer = ReplayBuffer(capacity=3, seed=0)
    push_numbered(buffer, 5)
    _, _, rewards, _, _ = buffer.sample(3)
    assert sorted(rewards.tolist()) == [2.0, 3.0, 4.0]


def test_sample_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=20, seed=0)
    push_numbered(buffer, 20)
    states, actions, rewards, next_states, dones = buffer.sample(8)
    assert states.shape == (8, 11) and states.dtype == np.float32
    assert next_states.shape == (8, 11) and next_states.dtype == np.float32
    assert actions.shape == (8,) and actions.dtype == np.int64
    assert rewards.shape == (8,) and rewards.dtype == np.float32
    assert dones.shape == (8,) and dones.dtype == np.float32


def test_sampled_rows_belong_to_the_same_transition():
    buffer = ReplayBuffer(capacity=50, seed=1)
    push_numbered(buffer, 50)
    states, actions, rewards, next_states, dones = buffer.sample(20)
    np.testing.assert_array_equal(states[:, 0], rewards)
    np.testing.assert_array_equal(next_states[:, 0], rewards + 1)
    np.testing.assert_array_equal(actions, (rewards % 3).astype(np.int64))
    np.testing.assert_array_equal(dones, (rewards % 2 == 0).astype(np.float32))


def test_sample_has_no_repetitions():
    buffer = ReplayBuffer(capacity=30, seed=2)
    push_numbered(buffer, 30)
    _, _, rewards, _, _ = buffer.sample(30)
    assert len(set(rewards.tolist())) == 30


def test_same_seed_gives_same_batch():
    first = ReplayBuffer(capacity=40, seed=7)
    second = ReplayBuffer(capacity=40, seed=7)
    push_numbered(first, 40)
    push_numbered(second, 40)
    np.testing.assert_array_equal(first.sample(10)[2], second.sample(10)[2])


def test_sampling_more_than_stored_raises():
    buffer = ReplayBuffer(capacity=10)
    push_numbered(buffer, 4)
    with pytest.raises(ValueError):
        buffer.sample(5)


def test_non_positive_capacity_is_rejected():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_push_stores_a_copy_of_the_states():
    buffer = ReplayBuffer(capacity=2, seed=0)
    state = np.zeros(11, dtype=np.float32)
    next_state = np.ones(11, dtype=np.float32)
    buffer.push(state, 0, 1.0, next_state, False)
    state[:] = 99
    next_state[:] = 99
    states, _, _, next_states, _ = buffer.sample(1)
    assert states.max() == 0.0
    assert next_states.max() == 1.0
