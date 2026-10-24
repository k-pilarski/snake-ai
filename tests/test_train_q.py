import subprocess
import sys

from snake_ai.train_q import train


def test_train_returns_one_score_per_episode():
    agent, scores = train(episodes=20, seed=0, log_every=0)
    assert len(scores) == 20
    assert all(isinstance(s, int) and s >= 0 for s in scores)
    assert len(agent.q_table) > 0


def test_train_decays_epsilon():
    agent, _ = train(episodes=20, seed=0, log_every=0)
    assert agent.epsilon < 1.0


def test_train_is_reproducible_with_same_seed():
    _, first = train(episodes=30, seed=5, log_every=0)
    _, second = train(episodes=30, seed=5, log_every=0)
    assert first == second


def test_train_logs_running_mean(capsys):
    train(episodes=10, seed=0, log_every=5)
    output = capsys.readouterr().out
    assert output.count("mean score") == 2


def test_training_does_not_import_pygame():
    code = "import sys, snake_ai.train_q; assert 'pygame' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
