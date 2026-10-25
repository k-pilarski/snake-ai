# snake-ai

A Snake game played by an AI agent that learns from scratch using reinforcement learning (Q-learning, then DQN with PyTorch).

## Status

Milestones M0-M2 are done: the game environment, a tabular Q-learning agent and its baseline results. Work continues on the DQN agent. See the [project board](https://github.com/users/k-pilarski/projects/5) and [milestones](../../milestones) for current progress.

## Roadmap

- [x] M0: Project setup
- [x] M1: Snake game environment
- [x] M2: Tabular Q-learning agent
- [ ] M3: Deep Q-Network (DQN) agent
- [ ] M4: Visualization and metrics
- [ ] M5: Documentation and release

## Results so far

The tabular Q-learning agent reaches a mean score of about 24.7 in greedy evaluation (a random policy scores 0.06). Details are in [docs/q-learning-baseline.md](docs/q-learning-baseline.md).

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
    pytest

Play Snake yourself with the keyboard:

    python -m snake_ai.play_manual

Train the Q-learning agent and run the benchmark:

    python -m snake_ai.train_q --episodes 3000
    python -m snake_ai.benchmark_q

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
