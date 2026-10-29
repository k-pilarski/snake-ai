# snake-ai

A Snake game played by an AI agent that learns from scratch using reinforcement learning (Q-learning, then DQN with PyTorch).

## Status

Milestones M0-M3 are done: the game environment, a tabular Q-learning agent, a DQN agent and a comparison of the two. Work continues on visualization, metrics and the release. See the [project board](https://github.com/users/k-pilarski/projects/5) and [milestones](../../milestones) for current progress.

## Roadmap

- [x] M0: Project setup
- [x] M1: Snake game environment
- [x] M2: Tabular Q-learning agent
- [x] M3: Deep Q-Network (DQN) agent
- [ ] M4: Visualization and metrics
- [ ] M5: Documentation and release

## Results so far

Greedy evaluation, 1000 training episodes, mean over five seeds (a random policy scores 0.06):

| Agent | Mean score |
|---|---|
| Tabular Q-learning | 26.55 |
| DQN (best saved model) | 28.03 |

DQN scored higher on all five seeds, but the gap is small and the agents see the same 11 features. Details, hyperparameters and limitations are in [docs/q-learning-vs-dqn.md](docs/q-learning-vs-dqn.md) and [docs/q-learning-baseline.md](docs/q-learning-baseline.md).

## Tech stack

Python 3.11, pygame, PyTorch, NumPy, matplotlib. Linting with ruff, tests with pytest.

## Getting started

Requires Python 3.11 or newer.

Create and activate a virtual environment:

    python -m venv .venv

    # Windows (Git Bash)
    source .venv/Scripts/activate

    # Linux / macOS
    source .venv/bin/activate

Install the project with development tools:

    pip install -e ".[dev]"

Run the checks:

    ruff check .
    python -m pytest

Play Snake yourself with the keyboard:

    python -m snake_ai.play_manual

Train the agents and compare them:

    python -m snake_ai.train_q --episodes 3000
    python -m snake_ai.train_dqn --episodes 1000
    python -m snake_ai.compare

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
