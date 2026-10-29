"""Compare tabular Q-learning and DQN: python -m snake_ai.compare

For every seed both agents train for the same number of episodes and are then
evaluated greedily on the same environment seed. Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
from pathlib import Path

import torch

from snake_ai.benchmark_q import evaluate, random_baseline
from snake_ai.dqn_agent import DQNAgent
from snake_ai.train_dqn import train as train_dqn
from snake_ai.train_q import train as train_q


def load_dqn(path: str | Path) -> DQNAgent:
    """Load a saved DQN model into an agent that always acts greedily."""
    agent = DQNAgent(epsilon=0.0)
    agent.model.load_state_dict(torch.load(path))
    return agent


def _range(values: list[float]) -> str:
    return f"{min(values):.2f}-{max(values):.2f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare Q-learning and DQN.")
    parser.add_argument("--episodes", type=int, default=1000, help="training episodes per agent")
    parser.add_argument("--seeds", type=int, default=5, help="number of training seeds")
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args(argv)

    print(f"Training: {args.episodes} episodes per agent and seed.")
    print(f"Evaluation: {args.eval_episodes} greedy episodes on a separate environment seed.")
    print("DQN is evaluated as the best saved model (rolling mean) and as the final model.")
    print()
    print("| Seed | Q mean | Q best | DQN mean | DQN best | DQN final mean |")
    print("|---|---|---|---|---|---|")

    q_means: list[float] = []
    dqn_means: list[float] = []
    final_means: list[float] = []
    q_bests: list[int] = []
    dqn_bests: list[int] = []

    for seed in range(args.seeds):
        eval_seed = 1000 + seed

        q_agent, _ = train_q(episodes=args.episodes, seed=seed, log_every=0)
        q_scores = evaluate(q_agent, episodes=args.eval_episodes, seed=eval_seed)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dqn_best.pt"
            dqn_agent, _ = train_dqn(
                episodes=args.episodes, seed=seed, log_every=0, save_path=path
            )
            final_scores = evaluate(dqn_agent, episodes=args.eval_episodes, seed=eval_seed)
            best_agent = load_dqn(path)
            best_scores = evaluate(best_agent, episodes=args.eval_episodes, seed=eval_seed)

        q_means.append(statistics.mean(q_scores))
        q_bests.append(max(q_scores))
        dqn_means.append(statistics.mean(best_scores))
        dqn_bests.append(max(best_scores))
        final_means.append(statistics.mean(final_scores))
        print(
            f"| {seed} | {q_means[-1]:.2f} | {q_bests[-1]} | {dqn_means[-1]:.2f} | "
            f"{dqn_bests[-1]} | {final_means[-1]:.2f} |",
            flush=True,
        )

    baseline = random_baseline(episodes=args.eval_episodes)
    print()
    print(
        f"Q-learning: mean of eval means {statistics.mean(q_means):.2f} "
        f"(range {_range(q_means)}), best single episode {max(q_bests)}."
    )
    print(
        f"DQN (best saved model): mean of eval means {statistics.mean(dqn_means):.2f} "
        f"(range {_range(dqn_means)}), best single episode {max(dqn_bests)}."
    )
    print(
        f"DQN (final model): mean of eval means {statistics.mean(final_means):.2f} "
        f"(range {_range(final_means)})."
    )
    print(
        f"Random policy: mean {statistics.mean(baseline):.2f}, best {max(baseline)} "
        f"over {args.eval_episodes} episodes."
    )


if __name__ == "__main__":
    main()
