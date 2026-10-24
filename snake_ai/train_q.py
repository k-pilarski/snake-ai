"""Train the tabular Q-learning agent: python -m snake_ai.train_q"""

from __future__ import annotations

import argparse

from snake_ai.env import SnakeEnv
from snake_ai.q_agent import QAgent


def train(
    episodes: int = 3000,
    seed: int = 0,
    log_every: int = 250,
    agent: QAgent | None = None,
    env: SnakeEnv | None = None,
) -> tuple[QAgent, list[int]]:
    """Run `episodes` training episodes without rendering.

    Returns the trained agent and the score of every episode.
    """
    env = env if env is not None else SnakeEnv(seed=seed)
    agent = agent if agent is not None else QAgent(seed=seed + 1)
    scores: list[int] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        info: dict = {"score": 0}
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
        scores.append(info["score"])

        if log_every > 0 and episode % log_every == 0:
            recent = scores[-log_every:]
            print(
                f"episode {episode:6d}  "
                f"mean score {sum(recent) / len(recent):6.2f}  "
                f"best {max(scores):3d}  "
                f"epsilon {agent.epsilon:.3f}  "
                f"states {len(agent.q_table)}"
            )

    return agent, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning Snake agent.")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()
    train(episodes=args.episodes, seed=args.seed, log_every=args.log_every)


if __name__ == "__main__":
    main()
