"""Metrics, baselines and calibration for the outcome models (numpy and pandas only).

Probabilities are arrays of shape (rows, classes); labels are class indexes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-15


def log_loss(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log probability of the true class."""
    picked = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, EPSILON, 1.0))))


def brier(y: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between the probabilities and the one-hot labels, summed over
    classes (for two classes this is twice the usual binary Brier score)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def frequency_baseline(
    train_groups: pd.DataFrame,
    train_y: np.ndarray,
    test_groups: pd.DataFrame,
    n_classes: int,
    prior_weight: float = 10.0,
) -> np.ndarray:
    """Class frequencies per group (e.g. per count, per pitch type) learned on train.

    Each group's frequencies are smoothed toward the overall frequencies by `prior_weight`
    rows; groups not seen in training get the overall frequencies. With no group columns,
    this is the prior-only baseline.
    """
    overall = np.bincount(train_y, minlength=n_classes) / len(train_y)
    if train_groups.shape[1] == 0:
        return np.tile(overall, (len(test_groups), 1))
    keys = list(train_groups.columns)
    # Group on strings so train and test keys compare equal whatever their dtypes
    train_keys = {k: train_groups[k].astype(str).to_numpy() for k in keys}
    counts = (
        pd.DataFrame({**train_keys, "y": train_y})
        .groupby([*keys, "y"])
        .size()
        .unstack("y", fill_value=0)
        .reindex(columns=range(n_classes), fill_value=0)
    )
    totals = counts.sum(axis=1).to_numpy()[:, None]
    smoothed = (counts.to_numpy() + prior_weight * overall) / (totals + prior_weight)
    table = pd.DataFrame(smoothed, index=counts.index)
    test_keys = test_groups[keys].astype(str)
    index = pd.Index(test_keys[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(test_keys)
    looked_up = table.reindex(index)
    result = looked_up.to_numpy().copy()
    missing = np.isnan(result).any(axis=1)
    result[missing] = overall
    return result


def calibration_table(y_is_class: np.ndarray, p_class: np.ndarray, bins: int = 20) -> pd.DataFrame:
    """Reliability table for one class: equal-count bins of predicted probability, with
    the mean prediction and the observed rate in each."""
    order = np.argsort(p_class, kind="stable")
    groups = np.array_split(order, bins)
    rows = [
        {
            "bin": i,
            "rows": len(g),
            "mean_predicted": float(p_class[g].mean()),
            "observed_rate": float(y_is_class[g].mean()),
        }
        for i, g in enumerate(groups)
        if len(g)
    ]
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Row-weighted mean absolute gap between prediction and outcome across bins."""
    gaps = (table["mean_predicted"] - table["observed_rate"]).abs()
    return float(np.average(gaps, weights=table["rows"]))
