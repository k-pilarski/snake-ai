import torch

from snake_ai.dqn_model import DQNModel
from snake_ai.env import NUM_ACTIONS, SnakeEnv
from snake_ai.state import STATE_SIZE


def test_output_shape_for_batch():
    model = DQNModel()
    output = model(torch.zeros(32, STATE_SIZE))
    assert output.shape == (32, NUM_ACTIONS)
    assert output.dtype == torch.float32


def test_output_shape_for_single_state():
    model = DQNModel()
    output = model(torch.zeros(1, STATE_SIZE))
    assert output.shape == (1, 3)


def test_architecture_is_11_256_3():
    model = DQNModel()
    linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear_layers] == [(11, 256), (256, 3)]
    assert sum(p.numel() for p in model.parameters()) == 3843


def test_accepts_state_from_environment():
    state = SnakeEnv(seed=0).reset()
    output = DQNModel()(torch.from_numpy(state).unsqueeze(0))
    assert output.shape == (1, 3)


def test_gradients_flow_to_all_parameters():
    model = DQNModel()
    loss = model(torch.ones(4, STATE_SIZE)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
