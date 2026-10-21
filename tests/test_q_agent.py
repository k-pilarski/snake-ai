import numpy as np
import pytest

from snake_ai.env import NUM_ACTIONS
from snake_ai.q_agent import QAgent

S1 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32)
S2 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)


def test_new_state_starts_with_zeros():
    agent = QAgent(seed=0)
    values = agent.q_values(S1)
    assert values.shape == (NUM_ACTIONS,)
    assert not values.any()


def test_update_rule_for_non_terminal_transition():
    agent = QAgent(alpha=0.5, gamma=0.9, seed=0)
    agent.q_values(S2)[1] = 4.0
    agent.update(S1, 0, 10.0, S2, done=False)
    # 0 + 0.5 * (10 + 0.9 * 4.0 - 0) = 6.8
    assert agent.q_values(S1)[0] == pytest.approx(6.8)


def test_update_ignores_future_for_terminal_transition():
    agent = QAgent(alpha=0.5, gamma=0.9, seed=0)
    agent.q_values(S2)[1] = 4.0
    agent.update(S1, 2, -10.0, S2, done=True)
    assert agent.q_values(S1)[2] == pytest.approx(-5.0)


def test_greedy_action_picks_highest_value():
    agent = QAgent(epsilon=0.0, seed=0)
    agent.q_values(S1)[:] = [1.0, 3.0, 2.0]
    assert all(agent.select_action(S1) == 1 for _ in range(20))


def test_ties_are_broken_randomly():
    agent = QAgent(epsilon=0.0, seed=0)
    actions = {agent.select_action(S1) for _ in range(100)}
    assert actions == {0, 1, 2}


def test_full_exploration_uses_all_actions():
    agent = QAgent(epsilon=1.0, seed=0)
    agent.q_values(S1)[:] = [1.0, 3.0, 2.0]
    actions = {agent.select_action(S1) for _ in range(200)}
    assert actions == {0, 1, 2}


def test_epsilon_decays_to_minimum():
    agent = QAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_same_seed_gives_same_actions():
    a = QAgent(epsilon=1.0, seed=3)
    b = QAgent(epsilon=1.0, seed=3)
    assert [a.select_action(S1) for _ in range(30)] == [b.select_action(S1) for _ in range(30)]


def test_states_are_stored_separately():
    agent = QAgent(seed=0)
    agent.update(S1, 0, 10.0, S2, done=True)
    assert agent.q_values(S1)[0] != 0.0
    assert not agent.q_values(S2).any()
    assert len(agent.q_table) == 2
