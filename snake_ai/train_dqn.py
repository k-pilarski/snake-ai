"""Train the DQN agent: python -m snake_ai.train_dqn"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from snake_ai.dqn_agent import DQNAgent
from snake_ai.env import SnakeEnv

DEFAULT_SAVE_PATH = "models/dqn_best.pt"


def save_model(agent: DQNAgent, path: str | Path) -> None:
    """Save the network weights of the agent, creating the directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.model.state_dict(), path)


def train(
    episodes: int = 1000,
    seed: int = 0,
    log_every: int = 50,
    save_path: str | Path | None = DEFAULT_SAVE_PATH,
    window: int = 100,
    agent: DQNAgent | None = None,
    env: SnakeEnv | None = None,
) -> tuple[DQNAgent, list[int]]:
    """Train without rendering and return the agent and the score of every episode.

    The model is saved to `save_path` whenever the mean score over the last
    `window` episodes beats its previous best. Pass save_path=None to disable saving.
    """
    env = env if env is not None else SnakeEnv(seed=seed)
    agent = agent if agent is not None else DQNAgent(seed=seed + 1)
    window = max(1, min(window, episodes))
    scores: list[int] = []
    losses: list[float] = []
    best_mean = float("-inf")

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        info: dict = {"score": 0}
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None and log_every > 0:
                losses.append(loss)
            state = next_state
        agent.decay_epsilon()
        scores.append(info["score"])

        if save_path is not None and episode >= window:
            mean = sum(scores[-window:]) / window
            if mean > best_mean:
                best_mean = mean
                save_model(agent, save_path)

        if log_every > 0 and episode % log_every == 0:
            recent = scores[-log_every:]
            mean_loss = sum(losses) / len(losses) if losses else float("nan")
            losses.clear()
            print(
                f"episode {episode:6d}  "
                f"mean score {sum(recent) / len(recent):6.2f}  "
                f"best {max(scores):3d}  "
                f"epsilon {agent.epsilon:.3f}  "
                f"loss {mean_loss:8.3f}"
            )

    return agent, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN Snake agent.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        seed=args.seed,
        log_every=args.log_every,
        save_path=args.save_path,
        window=args.window,
    )


if __name__ == "__main__":
    main()
