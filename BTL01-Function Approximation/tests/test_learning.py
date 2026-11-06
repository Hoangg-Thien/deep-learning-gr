import numpy as np
import pytest
import torch

from src.data import generate_splits
from src.metrics import regression_metrics
from src.model import MLP
from src.training import fit, load_checkpoint, predict, seed_everything


@pytest.mark.parametrize('widths', [[1], [10], [10, 10, 10, 10], [8, 4]])
@pytest.mark.parametrize('activation', ['tanh', 'sigmoid', 'relu'])
def test_model_shape_forward_backward(widths, activation):
    seed_everything(1)
    model = MLP(widths, activation)
    x = torch.randn(13, 2)
    result = model(x)
    assert result.shape == (13, 1)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_metrics_against_manual_values_and_no_broadcast():
    actual = regression_metrics(np.array([[0.], [2.]]), np.array([[1.], [0.]]))
    assert actual == pytest.approx({'mse': 2.5, 'mae': 1.5, 'rmse': np.sqrt(2.5)})
    with pytest.raises(ValueError):
        regression_metrics(np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(ValueError):
        regression_metrics(np.array([[np.nan]]), np.zeros((1, 1)))
    with pytest.raises(ValueError):
        regression_metrics(np.zeros((0, 1)), np.zeros((0, 1)))


def small_config():
    return dict(hidden_sizes=[10], activation='tanh', optimizer='adam', lr=0.01,
                batch_size=64, epochs=90, patience=30, min_delta=0., seed=11, device='cpu', threads=1)


def test_training_checkpoint_reproducibility_and_no_validation_grad(tmp_path):
    splits = generate_splits(200, 80, 100, 5)
    a = fit(splits['train'], splits['validation'], small_config(), tmp_path / 'a.pt')
    b = fit(splits['train'], splits['validation'], small_config(), tmp_path / 'b.pt')
    assert a['history'] == b['history']
    losses = [row['validation_loss'] for row in a['history']]
    assert a['best_validation_loss'] == min(losses)
    assert a['best_epoch'] == int(np.argmin(losses)) + 1
    assert losses[-1] < losses[0]
    loaded, metadata = load_checkpoint(tmp_path / 'a.pt')
    x, y = splits['validation']
    np.testing.assert_array_equal(predict(a['model'], x), predict(loaded, x))
    assert regression_metrics(y, predict(loaded, x))['mse'] == pytest.approx(min(losses), rel=1e-5)
    assert metadata['best_epoch'] == a['best_epoch']
    assert all(p.grad is None for p in loaded.parameters())
    np.testing.assert_array_equal(predict(a['model'], x), predict(b['model'], x))


def test_early_stopping_restores_best_not_last(tmp_path):
    splits = generate_splits(100, 50, 100, 1)
    config = small_config() | {'lr': 0., 'patience': 3, 'epochs': 20}
    result = fit(splits['train'], splits['validation'], config, tmp_path / 'early.pt')
    assert result['epochs_ran'] == 4
    assert result['best_epoch'] == 1
    assert result['stopped_early']


@pytest.mark.parametrize('widths,activation', [([], 'tanh'), ([0], 'tanh'), ([4], 'unknown')])
def test_invalid_architecture_rejected(widths, activation):
    with pytest.raises(ValueError):
        MLP(widths, activation)
