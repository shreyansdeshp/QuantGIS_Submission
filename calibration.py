"""
Calibration methods shared between train.py (fit on OOF) and model.py
(apply at serve time). Both are small enough to be plain NumPy at serve
time - no sklearn calibration classes, no per-request Python object
overhead beyond a handful of array ops.

Beta calibration (approve, grounded): a 3-parameter generalisation of
Platt scaling - logit(p_cal) = a*log(p_raw) - b*log(1-p_raw) + c - that
handles the skewed, already-stacked probabilities these heads produce
better than a plain sigmoid(a*z+b) can. Reference: Kull, Filho & Flach,
"Beta calibration: a well-founded and easily implemented improvement on
logistic calibration for binary classifiers" (2017).

Diagonal vector scaling (route): a 10-parameter generalisation of scalar
temperature scaling - softmax(a*log_p + b) with a, b in R^5 - that can
correct a per-class bias (e.g. "billing" systematically overconfident)
that a single shared temperature cannot.
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax


def fit_beta_calibration(p_raw: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    p = np.clip(p_raw, 1e-7, 1.0 - 1e-7)
    log_p = np.log(p)
    log_1mp = np.log(1.0 - p)
    y = y.astype(np.float64)

    def loss(params):
        a, b, c = params
        z = a * log_p - b * log_1mp + c
        p_cal = np.clip(expit(z), 1e-12, 1.0 - 1e-12)
        nll = -np.mean(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))
        brier = np.mean((p_cal - y) ** 2)
        return nll + 0.5 * brier

    res = minimize(loss, x0=[1.0, 1.0, 0.0],
                    bounds=[(0.01, 10.0), (0.01, 10.0), (-5.0, 5.0)],
                    method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def apply_beta_calibration(p_raw, a: float, b: float, c: float):
    p = np.clip(p_raw, 1e-7, 1.0 - 1e-7)
    z = a * np.log(p) - b * np.log(1.0 - p) + c
    return expit(z)


def fit_diagonal_scaling(proba: np.ndarray, y: np.ndarray, n_classes: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """proba: (N, K) softmax probabilities from the uncalibrated head.
    Fits per-class a_k, b_k on log(proba) to minimise NLL + 0.5*multiclass
    Brier, generalising the scalar-temperature fit already used for route."""
    log_p = np.log(np.clip(proba, 1e-9, 1.0))
    n, k = log_p.shape
    y_onehot = np.zeros((n, k))
    y_onehot[np.arange(n), y] = 1.0

    def unpack(params):
        return params[:k], params[k:]

    def loss(params):
        a, b = unpack(params)
        z = a[None, :] * log_p + b[None, :]
        p = softmax(z, axis=1)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        nll = -np.mean(np.sum(y_onehot * np.log(p), axis=1))
        brier = np.mean(np.sum((p - y_onehot) ** 2, axis=1))
        return nll + 0.5 * brier

    x0 = np.concatenate([np.ones(k), np.zeros(k)])
    bounds = [(0.1, 10.0)] * k + [(-5.0, 5.0)] * k
    res = minimize(loss, x0, bounds=bounds, method="L-BFGS-B")
    a, b = unpack(res.x)
    return a, b


def apply_diagonal_scaling(proba: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    log_p = np.log(np.clip(proba, 1e-9, 1.0))
    z = a[None, :] * log_p + b[None, :] if proba.ndim == 2 else a * log_p + b
    return softmax(z, axis=-1)
