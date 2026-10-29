# Q-learning baseline

This document records what the tabular Q-learning agent achieves. It is the
baseline that the DQN agent will be compared against.

## Setup

- Environment: 20x20 grid, solid walls, three relative actions (straight, turn right, turn left).
- Rewards: +10 for food, -10 for death or timeout, 0 otherwise.
- Timeout: an episode ends after 100 * snake length steps without eating.
- State: 11 binary features (danger straight/right/left, heading one-hot, food direction).
  The state space contains at most 256 distinct states.
- Agent: tabular Q-learning, alpha = 0.1, gamma = 0.9, epsilon decaying from 1.0 to 0.01
  with a factor of 0.995 per episode.
- Training: 3000 episodes per seed, seeds 0-4.
- Evaluation: after training, 100 greedy episodes (epsilon = 0, no learning) on a
  separate environment seed (1000 + training seed).

Score is the number of foods eaten in an episode.

## Results

| Seed | Train mean | Train best | Eval mean | Eval best | States |
|---|---|---|---|---|---|
| 0 | 22.27 | 60 | 24.96 | 56 | 255 |
| 1 | 21.80 | 58 | 24.84 | 51 | 256 |
| 2 | 22.36 | 64 | 25.53 | 51 | 255 |
| 3 | 20.07 | 56 | 23.85 | 48 | 255 |
| 4 | 21.06 | 57 | 24.29 | 60 | 255 |

- Train mean is the mean score over the last 500 training episodes (epsilon at its minimum).
- Train best is the best single episode of the whole training run.
- Eval mean and Eval best come from the greedy evaluation.
- States is the number of states in the Q-table after training.

Summary:

- Q-learning (greedy): mean of the evaluation means is **24.69** (range 23.85-25.53),
  best single evaluation episode **60**.
- Random policy: mean **0.06**, best 2 over 100 episodes.

## Observations

- The agent clearly learns: it scores several hundred times more than a random policy,
  and the result is consistent across seeds (the evaluation means differ by about one point).
- The best single episode varies much more than the mean between seeds, so the mean is
  the figure to compare against.
- In an earlier single run (seed 0) the mean score rose from 1.4 after 250 episodes to
  about 22 after 1000 episodes and then stayed at 21-23 until episode 3000.
- Evaluation scores are higher than the late training scores. A likely reason is that the
  training runs still take 1% random actions, which are costly for a long snake. This was
  not tested separately.
- The Q-table covers nearly the whole state space (255-256 of at most 256 states), so
  more training would not add new information.

## Limitations

The plateau is probably caused by the state representation, not by the amount of training.
The agent sees only the three cells next to its head, its heading and the direction of the
food. It does not know the length of the snake, the position of its tail or whether it is
closing itself into a dead end. A likely consequence is that a long snake sometimes traps
itself, but the cause of individual deaths was not measured.

The DQN agent will receive the same 11 features, so it has no more information than the
Q-table. It may reach a similar score rather than a clearly better one. A richer state
(more lookahead, free space, tail position or the whole grid) would be needed for a real
improvement.

## Reproduce

Training run with logging:

    python -m snake_ai.train_q --episodes 3000 --seed 0

Full benchmark (the table above):

    python -m snake_ai.benchmark_q

## Note on training length

The table above uses 3000 training episodes per seed. For an equal-budget comparison with the DQN agent, [q-learning-vs-dqn.md](q-learning-vs-dqn.md) uses 1000 episodes with the same seeds, and the Q-learning evaluation means there are higher (mean of eval means 26.55 instead of 24.69). Training is seeded, so the first 1000 episodes are identical in both runs. The cause of the lower scores after longer training was not investigated.
