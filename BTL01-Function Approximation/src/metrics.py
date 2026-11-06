"""Regression metrics in the original density units; no broadcasting."""
import numpy as np


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError('Metrics require equal nonempty shapes')
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('Metrics require finite values')
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    return dict(mse=mse, mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(mse)))
