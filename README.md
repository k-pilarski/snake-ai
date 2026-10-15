# snake-ai

A Snake game played by an AI agent that learns from scratch using reinforcement learning (Q-learning, then DQN with PyTorch).

## Status

Project setup in progress. See the [project board](https://github.com/users/k-pilarski/projects/5) and [milestones](../../milestones) for current progress.

## Roadmap

- [ ] M0: Project setup
- [ ] M1: Snake game environment
- [ ] M2: Tabular Q-learning agent
- [ ] M3: Deep Q-Network (DQN) agent
- [ ] M4: Visualization and metrics
- [ ] M5: Documentation and release

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
