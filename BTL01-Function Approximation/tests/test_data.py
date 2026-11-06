import numpy as np
import pytest

from src.data import COVARIANCE, gaussian_2d, generate_splits, load_csv, save_csv, validate_data


def test_gaussian_at_mean_and_symmetry():
    peak = 1 / (2 * np.pi * np.sqrt(0.35))
    assert gaussian_2d(np.array([[0., 0.]]))[0, 0] == pytest.approx(peak)
    x = np.array([[0.3, -1.2], [-0.3, 1.2]])
    np.testing.assert_allclose(gaussian_2d(x)[0], gaussian_2d(x)[1])
    delta = x[0]
    expected = peak * np.exp(-0.5 * delta @ np.linalg.inv(COVARIANCE) @ delta)
    assert gaussian_2d(x)[0, 0] == pytest.approx(expected)


def test_gaussian_integrates_to_one():
    a = np.linspace(-6, 6, 501)
    b = np.linspace(-12, 12, 501)
    xx, yy = np.meshgrid(a, b)
    z = gaussian_2d(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    integral = np.trapezoid(np.trapezoid(z, a, axis=1), b)
    assert integral == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize('cov', [[[1, 2], [2, 1]], [[1, 1], [0, 1]], [[1, 0], [0, 0]]])
def test_invalid_covariance_rejected(cov):
    with pytest.raises(ValueError):
        gaussian_2d(np.zeros((2, 2)), covariance=cov)


def test_generation_shape_range_seed_and_independence(tmp_path):
    splits = generate_splits(2000, 400, 100, seed=2026)
    again = generate_splits(2000, 400, 100, seed=2026)
    other = generate_splits(2000, 400, 100, seed=2027)
    for name, n in [('train', 2000), ('validation', 400), ('test', 100)]:
        x, y = splits[name]
        assert x.shape == (n, 2) and y.shape == (n, 1)
        validate_data(x, y)
        np.testing.assert_array_equal(x, again[name][0])
        assert not np.array_equal(x, other[name][0])
        save_csv(tmp_path / f'{name}.csv', x, y)
        actual_x, actual_y = load_csv(tmp_path / f'{name}.csv')
        np.testing.assert_allclose(actual_x, x, rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(actual_y, y, rtol=1e-14, atol=1e-15)
    sets = [set(map(tuple, splits[name][0])) for name in splits]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))


@pytest.mark.parametrize('x,y', [(np.zeros((2, 3)), np.zeros((2, 1))),
                                (np.array([[3., 0.]]), np.zeros((1, 1))),
                                (np.array([[np.nan, 0.]]), np.zeros((1, 1))),
                                (np.zeros((1, 2)), np.array([[-1.]]))])
def test_invalid_data_rejected(x, y):
    with pytest.raises(ValueError):
        validate_data(x, y)


def test_invalid_sample_count_rejected():
    with pytest.raises(ValueError):
        generate_splits(0, 100, 100, 1)
