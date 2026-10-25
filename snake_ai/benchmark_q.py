"""Measure the Q-learning baseline: python -m snake_ai.benchmark_q

Trains one agent per seed, evaluates it greedily (epsilon = 0, no learning) on
fresh episodes and compares it with a random policy. Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import random
import statistics

from snake_ai.env import NUM_ACTIONS, SnakeEnv
from snake_ai.q_agent import QAgent
from snake_ai.train_q import train

TRAIN_WINDOW = 500


def evaluate(agent: QAgent, episodes: int = 100, seed: int = 1000) -> list[int]:
    """Play greedy episodes without learning and return the score of each one."""
    env = SnakeEnv(seed=seed)
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0
    scores: list[int] = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        info: dict = {"score": 0}
        while not done:
            state, _, done, info = env.step(agent.select_action(state))
        scores.append(info["score"])
    agent.epsilon = previous_epsilon
    return scores


def random_baseline(episodes: int = 100, seed: int = 1000) -> list[int]:
    """Play episodes with uniformly random actions and return the scores."""
    env = SnakeEnv(seed=seed)
    rng = random.Random(seed)
    scores: list[int] = []
    for _ in range(episodes):
        env.reset()
        done = False
        info: dict = {"score": 0}
        while not done:
            _, _, done, info = env.step(rng.randrange(NUM_ACTIONS))
        scores.append(info["score"])
    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark the tabular Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=3000, help="training episodes per seed")
    parser.add_argument("--seeds", type=int, default=5, help="number of training seeds")
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args(argv)

    print(f"Training: {args.episodes} episodes per seed, mean over the last {TRAIN_WINDOW}.")
    print(f"Evaluation: {args.eval_episodes} greedy episodes on a separate environment seed.")
    print()
    print("| Seed | Train mean | Train best | Eval mean | Eval best | States |")
    print("|---|---|---|---|---|---|")

    eval_means: list[float] = []
    eval_bests: list[int] = []
    for seed in range(args.seeds):
        agent, scores = train(episodes=args.episodes, seed=seed, log_every=0)
        window = scores[-TRAIN_WINDOW:]
        eval_scores = evaluate(agent, episodes=args.eval_episodes, seed=1000 + seed)
        eval_means.append(statistics.mean(eval_scores))
        eval_bests.append(max(eval_scores))
        print(
            f"| {seed} | {statistics.mean(window):.2f} | {max(scores)} | "
            f"{eval_means[-1]:.2f} | {eval_bests[-1]} | {len(agent.q_table)} |"
        )

    baseline = random_baseline(episodes=args.eval_episodes)
    print()
    print(
        f"Q-learning (greedy): mean of eval means {statistics.mean(eval_means):.2f}, "
        f"best single episode {max(eval_bests)}."
    )
    print(
        f"Random policy: mean {statistics.mean(baseline):.2f}, best {max(baseline)} "
        f"over {args.eval_episodes} episodes."
    )


if __name__ == "__main__":
    main()
