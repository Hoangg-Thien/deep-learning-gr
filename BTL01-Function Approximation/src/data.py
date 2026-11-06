"""Gaussian density and fixed independent uniform-design datasets."""
from pathlib import Path
import numpy as np

MEAN = np.array([0., 0.])
COVARIANCE = np.array([[0.3, -0.5], [-0.5, 2.]])
LOW = np.array([-2., -4.])
HIGH = -LOW


def gaussian_2d(x, mean=MEAN, covariance=COVARIANCE):
    x, mean, cov = (np.asarray(v, dtype=np.float64) for v in (x, mean, covariance))
    if x.ndim != 2 or x.shape[1] != 2 or mean.shape != (2,) or cov.shape != (2, 2):
        raise ValueError('Expected x (N,2), mean (2,), covariance (2,2)')
    if not all(np.isfinite(v).all() for v in (x, mean, cov)):
        raise ValueError('Gaussian inputs must be finite')
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
        raise ValueError('Covariance must be symmetric')
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Covariance must be positive definite') from exc
    delta = x - mean
    quadratic = np.sum(delta * np.linalg.solve(cov, delta.T).T, axis=1)
    return (np.exp(-0.5 * quadratic) / (2 * np.pi * np.sqrt(np.linalg.det(cov))))[:, None]


def validate_data(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or y.shape != (len(x), 1):
        raise ValueError('Expected nonempty x (N,2) and y (N,1)')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Data must be finite')
    peak = float(gaussian_2d(MEAN[None, :])[0, 0])
    if np.any(x < LOW) or np.any(x > HIGH) or np.any(y < 0) or np.any(y > peak + 1e-12):
        raise ValueError('Data outside Gaussian domain/range')
    if not np.allclose(y, gaussian_2d(x), rtol=1e-10, atol=1e-14):
        raise ValueError('Targets disagree with official Gaussian')


def generate_splits(n_train=2000, n_validation=400, n_test=100, seed=2026):
    counts = [n_train, n_validation, n_test]
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
        raise ValueError('Sample counts must be positive integers')
    streams = np.random.SeedSequence(seed).spawn(3)
    splits = {}
    for name, n, stream in zip(('train', 'validation', 'test'), counts, streams):
        x = np.random.default_rng(stream).uniform(LOW, HIGH, size=(n, 2))
        y = gaussian_2d(x)
        validate_data(x, y)
        splits[name] = (x, y)
    row_sets = [set(map(tuple, x)) for x, _ in splits.values()]
    if any(row_sets[i] & row_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Duplicate observations across splits')
    return splits


def save_csv(path, x, y):
    validate_data(x, y)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([x, y]), delimiter=',', header='x1,x2,y', comments='', fmt='%.17g')


def load_csv(path):
    array = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if array.shape[1] != 3:
        raise ValueError('CSV must have exactly three columns')
    x, y = array[:, :2], array[:, 2:3]
    validate_data(x, y)
    return x, y
