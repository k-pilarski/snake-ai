# Q-learning vs DQN

This document compares the tabular Q-learning agent with the DQN agent under the same
conditions and records the hyperparameters and reward settings of both.

## Setup

Both agents play the same environment and see the same observation.

| Item | Value |
|---|---|
| Grid | 20x20, solid walls |
| Actions | 3, relative to the heading: straight, turn right, turn left |
| Rewards | +10 for food, -10 for death or timeout, 0 otherwise |
| Timeout | the episode ends after 100 * snake length steps without eating |
| State | 11 binary features: danger straight/right/left, heading one-hot, food direction |
| Score | number of foods eaten in an episode |

### Q-learning

| Hyperparameter | Value |
|---|---|
| Representation | Q-table keyed by the 11-feature state, 3 values per state |
| Update | Q(s,a) += alpha * (r + gamma * max Q(s') - Q(s,a)), no future term when done |
| Learning rate (alpha) | 0.1 |
| Discount (gamma) | 0.9 |
| Epsilon | 1.0 decaying by 0.995 per episode, minimum 0.01 |
| Ties | broken at random |

### DQN

| Hyperparameter | Value |
|---|---|
| Network | MLP 11 -> 256 -> 3, ReLU hidden layer, linear output (3843 parameters) |
| Target | r + gamma * max Q_target(s'), just r when done |
| Loss and optimizer | MSE, Adam |
| Learning rate | 0.001 |
| Discount (gamma) | 0.9 |
| Epsilon | 1.0 decaying by 0.995 per episode, minimum 0.01 |
| Batch size | 64 |
| Replay buffer | 100000 transitions, random batches without repetition |
| Target network | copy of the model, synchronized every 100 learning steps |
| Learning | one learning step per game step, once the buffer holds a full batch |
| Saved model | weights saved when the mean score over the last 100 training episodes reaches a new best |

### Protocol

- Training: 1000 episodes per agent, seeds 0-4. The environment uses the seed, the agent uses seed + 1.
- Evaluation: after training, 100 greedy episodes (epsilon = 0, no learning) on an environment seeded with 1000 + seed. Both agents use the same evaluation seed.
- DQN is evaluated twice: as the best saved model and as the final model after training.

## Results

| Seed | Q mean | Q best | DQN mean | DQN best | DQN final mean | DQN - Q (mean) |
|---|---|---|---|---|---|---|
| 0 | 26.06 | 63 | 26.81 | 50 | 26.17 | +0.75 |
| 1 | 25.90 | 58 | 28.59 | 63 | 23.68 | +2.69 |
| 2 | 27.31 | 52 | 28.03 | 56 | 27.53 | +0.72 |
| 3 | 27.36 | 58 | 29.40 | 63 | 32.27 | +2.04 |
| 4 | 26.12 | 48 | 27.34 | 48 | 27.80 | +1.22 |

DQN mean and DQN best refer to the best saved model. The last column is the difference
between its mean and the Q-learning mean.

Summary (mean of the five evaluation means):

| Agent | Mean | Range over seeds | Best single episode |
|---|---|---|---|
| Q-learning | 26.55 | 25.90-27.36 | 63 |
| DQN, best saved model | 28.03 | 26.81-29.40 | 63 |
| DQN, final model | 27.49 | 23.68-32.27 | not recorded |
| Random policy | 0.06 | not applicable | 2 |

## Observations

- Both agents learn and score far above a random policy.
- The DQN best saved model scored higher than Q-learning on all five seeds, by 0.72 to 2.69
  points, on average by 1.48 points (about 5-6%).
- The gap is small. The ranges over seeds overlap, and every number is the mean of 100
  episodes whose scores vary widely (the best episodes are 48-63 while the means are around
  27). The noise of a single measurement is probably around one point, but it was not
  measured. A consistent sign over five seeds suggests a modest advantage for DQN, but it is
  not conclusive.
- The final DQN model varies more between seeds (23.68-32.27) than the saved one. It scored
  lower than the saved model on seeds 0, 1 and 2 and higher on seeds 3 and 4. The saved model
  mostly reduces the spread; its advantage in the mean (28.03 vs 27.49) is within the noise.
- The best single episode is 63 for both agents, so it does not separate them.
- DQN with seed 0 scored 26.81 both in the separate training run and in this comparison, so
  the results are reproducible.
- Q-learning trained for 3000 episodes (see [q-learning-baseline.md](q-learning-baseline.md))
  scored a mean of 24.69, lower than the 26.55 after 1000 episodes here. It was lower on all
  five seeds. Training is seeded, so the first 1000 episodes are identical in both runs.
  The cause was not investigated. This is why the numbers in the two documents differ.

## Limitations

- Both agents see the same 11 features, so the neural network has no more information than
  the Q-table. The small gap is consistent with both being limited by the state
  representation, but this was not tested, for example with a richer state.
- The comparison uses an equal budget of 1000 episodes. In an earlier DQN training run
  (seed 0) the training score was still rising at episode 1000, so longer DQN training could
  change the result. This was not tested.
- Five seeds, one evaluation environment seed per agent and 100 evaluation episodes give
  only a rough estimate of the difference.
- Each agent was run with one set of hyperparameters and neither was tuned.
- Training time was not measured. DQN performs a learning step for every game step, so it is
  expected to train more slowly than the Q-table, which only updates a dictionary entry.

## Reproduce

Full comparison (the table above):

    python -m snake_ai.compare

Single runs:

    python -m snake_ai.train_q --episodes 1000 --seed 0
    python -m snake_ai.train_dqn --episodes 1000 --seed 0
