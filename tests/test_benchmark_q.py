import subprocess
import sys

from snake_ai.benchmark_q import evaluate, main, random_baseline
from snake_ai.q_agent import QAgent


def test_evaluate_returns_scores_and_restores_epsilon():
    agent = QAgent(epsilon=0.5, seed=0)
    scores = evaluate(agent, episodes=5, seed=1)
    assert len(scores) == 5
    assert all(isinstance(s, int) and s >= 0 for s in scores)
    assert agent.epsilon == 0.5


def test_random_baseline_returns_scores():
    scores = random_baseline(episodes=5, seed=1)
    assert len(scores) == 5
    assert all(isinstance(s, int) and s >= 0 for s in scores)


def test_main_prints_table_and_baseline(capsys):
    main(["--episodes", "20", "--seeds", "2", "--eval-episodes", "5"])
    output = capsys.readouterr().out
    assert "| Seed |" in output
    assert "| 0 |" in output and "| 1 |" in output
    assert "Random policy" in output


def test_benchmark_does_not_import_pygame():
    code = "import sys, snake_ai.benchmark_q; assert 'pygame' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
