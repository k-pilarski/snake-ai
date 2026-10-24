import json

import numpy as np
import pytest

from snake_ai.q_agent import QAgent

S1 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32)
S2 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)


def make_trained_agent() -> QAgent:
    agent = QAgent(alpha=0.3, gamma=0.8, epsilon=0.4, seed=0)
    agent.update(S1, 0, 10.0, S2, done=False)
    agent.update(S2, 2, -10.0, S1, done=True)
    agent.update(S1, 1, 0.1, S2, done=False)
    return agent


def test_save_creates_missing_directory(tmp_path):
    path = tmp_path / "models" / "q.json"
    make_trained_agent().save(path)
    assert path.is_file()


def test_saved_file_is_plain_json(tmp_path):
    path = tmp_path / "q.json"
    make_trained_agent().save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"hyperparameters", "q_table"}
    assert len(data["q_table"]) == 2


def test_round_trip_restores_q_table_exactly(tmp_path):
    agent = make_trained_agent()
    path = tmp_path / "q.json"
    agent.save(path)
    loaded = QAgent.load(path)
    assert loaded.q_table.keys() == agent.q_table.keys()
    for key, values in agent.q_table.items():
        np.testing.assert_array_equal(loaded.q_table[key], values)


def test_round_trip_restores_hyperparameters(tmp_path):
    agent = make_trained_agent()
    path = tmp_path / "q.json"
    agent.save(path)
    loaded = QAgent.load(path)
    for name in ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay"):
        assert getattr(loaded, name) == getattr(agent, name)


def test_loaded_agent_behaves_identically(tmp_path):
    agent = make_trained_agent()
    agent.epsilon = 0.0
    path = tmp_path / "q.json"
    agent.save(path)
    original = QAgent.load(path, seed=11)
    loaded = QAgent.load(path, seed=11)
    states = [S1, S2, np.zeros(11, dtype=np.float32)]
    assert [original.select_action(s) for s in states * 10] == [
        loaded.select_action(s) for s in states * 10
    ]


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        QAgent.load(tmp_path / "missing.json")
