import subprocess
import sys

import torch

from snake_ai.compare import load_dqn, main
from snake_ai.dqn_agent import DQNAgent
from snake_ai.train_dqn import save_model


def test_load_dqn_restores_weights_and_acts_greedily(tmp_path):
    agent = DQNAgent(seed=0)
    path = tmp_path / "model.pt"
    save_model(agent, path)
    loaded = load_dqn(path)
    assert loaded.epsilon == 0.0
    assert all(
        torch.equal(a, b)
        for a, b in zip(agent.model.parameters(), loaded.model.parameters(), strict=True)
    )


def test_main_prints_comparison_table(capsys):
    main(["--episodes", "10", "--seeds", "2", "--eval-episodes", "3"])
    output = capsys.readouterr().out
    assert "| Seed |" in output
    assert "| 0 |" in output and "| 1 |" in output
    assert "Q-learning:" in output
    assert "DQN (best saved model)" in output
    assert "Random policy" in output


def test_compare_does_not_import_pygame():
    code = "import sys, snake_ai.compare; assert 'pygame' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
