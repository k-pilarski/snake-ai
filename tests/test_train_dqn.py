import subprocess
import sys

import torch

from snake_ai import train_dqn
from snake_ai.dqn_agent import DQNAgent
from snake_ai.dqn_model import DQNModel
from snake_ai.train_dqn import train


def small_agent(seed: int = 1) -> DQNAgent:
    return DQNAgent(batch_size=8, memory_capacity=500, seed=seed)


def count_improvements(scores: list[int], window: int) -> int:
    best = float("-inf")
    count = 0
    for end in range(window, len(scores) + 1):
        mean = sum(scores[end - window : end]) / window
        if mean > best:
            best = mean
            count += 1
    return count


def test_train_returns_one_score_per_episode_and_learns():
    agent, scores = train(episodes=10, seed=0, log_every=0, save_path=None, agent=small_agent())
    assert len(scores) == 10
    assert all(isinstance(s, int) and s >= 0 for s in scores)
    assert agent.learn_steps > 0
    assert agent.epsilon < 1.0


def test_best_model_is_saved_and_can_be_loaded(tmp_path):
    path = tmp_path / "models" / "best.pt"
    train(episodes=10, seed=0, log_every=0, save_path=path, window=5, agent=small_agent())
    assert path.is_file()
    model = DQNModel()
    model.load_state_dict(torch.load(path))


def test_model_is_saved_only_when_the_mean_improves(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(train_dqn, "save_model", lambda agent, path: calls.append(path))
    _, scores = train(
        episodes=12,
        seed=0,
        log_every=0,
        save_path=tmp_path / "best.pt",
        window=3,
        agent=small_agent(),
    )
    assert len(calls) == count_improvements(scores, 3)
    assert len(calls) >= 1


def test_train_is_reproducible_with_same_seed():
    _, first = train(episodes=12, seed=4, log_every=0, save_path=None)
    _, second = train(episodes=12, seed=4, log_every=0, save_path=None)
    assert first == second


def test_train_logs_progress(capsys):
    train(episodes=10, seed=0, log_every=5, save_path=None, agent=small_agent())
    output = capsys.readouterr().out
    assert output.count("mean score") == 2
    assert "loss" in output


def test_training_does_not_import_pygame():
    code = "import sys, snake_ai.train_dqn; assert 'pygame' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
