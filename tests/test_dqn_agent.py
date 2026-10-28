import numpy as np
import pytest
import torch

from snake_ai.dqn_agent import DQNAgent


def set_constant_q(model, values):
    """Make a model return the same Q-values for every state."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[2].bias.copy_(torch.tensor(values))


def fill_memory(agent: DQNAgent, count: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(count):
        agent.remember(
            rng.integers(0, 2, 11).astype(np.float32),
            i % 3,
            float(i % 5),
            rng.integers(0, 2, 11).astype(np.float32),
            i % 4 == 0,
        )


def same_parameters(first, second) -> bool:
    return all(
        torch.equal(a, b)
        for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )


def test_train_step_waits_for_enough_transitions():
    agent = DQNAgent(batch_size=8, seed=0)
    fill_memory(agent, 3)
    assert agent.train_step() is None
    assert agent.learn_steps == 0


def test_train_step_returns_loss_and_changes_the_model():
    agent = DQNAgent(batch_size=8, seed=0)
    fill_memory(agent, 20)
    before = [p.detach().clone() for p in agent.model.parameters()]
    loss = agent.train_step()
    assert isinstance(loss, float)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.model.parameters(), strict=True)
    )
    assert agent.learn_steps == 1


def test_bellman_targets():
    agent = DQNAgent(gamma=0.9, seed=0)
    set_constant_q(agent.target_model, [1.0, 2.0, 3.0])
    targets = agent.compute_targets(
        torch.tensor([5.0, 5.0]),
        torch.zeros(2, 11),
        torch.tensor([0.0, 1.0]),
    )
    # non-terminal: 5 + 0.9 * 3 = 7.7, terminal: just the reward
    assert targets.tolist() == pytest.approx([7.7, 5.0])


def test_loss_decreases_on_a_fixed_transition():
    agent = DQNAgent(learning_rate=0.01, batch_size=16, target_update_every=10, seed=0)
    state = np.zeros(11, dtype=np.float32)
    for _ in range(16):
        agent.remember(state, 1, 10.0, state, True)
    losses = [agent.train_step() for _ in range(200)]
    assert losses[-1] < losses[0]
    assert losses[-1] < 1.0


def test_greedy_action_picks_highest_q_value():
    agent = DQNAgent(epsilon=0.0, seed=0)
    set_constant_q(agent.model, [0.0, 5.0, 1.0])
    rng = np.random.default_rng(1)
    for _ in range(10):
        state = rng.integers(0, 2, 11).astype(np.float32)
        assert agent.select_action(state) == 1


def test_full_exploration_uses_all_actions():
    agent = DQNAgent(epsilon=1.0, seed=0)
    state = np.zeros(11, dtype=np.float32)
    assert {agent.select_action(state) for _ in range(100)} == {0, 1, 2}


def test_epsilon_decays_to_minimum():
    agent = DQNAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_target_network_is_synchronized_periodically():
    agent = DQNAgent(learning_rate=0.01, batch_size=4, target_update_every=2, seed=0)
    fill_memory(agent, 8)
    initial_target = [p.detach().clone() for p in agent.target_model.parameters()]

    agent.train_step()
    assert all(
        torch.equal(old, new)
        for old, new in zip(initial_target, agent.target_model.parameters(), strict=True)
    )
    assert not same_parameters(agent.model, agent.target_model)

    agent.train_step()
    assert same_parameters(agent.model, agent.target_model)


def test_same_seed_gives_same_initial_weights():
    first = DQNAgent(seed=3)
    second = DQNAgent(seed=3)
    assert same_parameters(first.model, second.model)
