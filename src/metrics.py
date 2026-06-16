"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _valid_pairs(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    data = pd.concat([y_true, y_pred], axis=1).dropna()
    return data.iloc[:, 0], data.iloc[:, 1]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error."""
    yt, yp = _valid_pairs(y_true, y_pred)
    return float(np.mean(np.abs(yp - yt))) if len(yt) else np.nan


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error."""
    yt, yp = _valid_pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((yp - yt) ** 2))) if len(yt) else np.nan


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Average forecast error, defined as forecast minus realized value."""
    yt, yp = _valid_pairs(y_true, y_pred)
    return float(np.mean(yp - yt)) if len(yt) else np.nan


def correlation(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Pearson correlation between realized and forecast values."""
    yt, yp = _valid_pairs(y_true, y_pred)
    return float(yt.corr(yp)) if len(yt) >= 2 else np.nan


def forecast_comparison_table(
    df: pd.DataFrame,
    target_col: str,
    forecast_cols: list[str],
) -> pd.DataFrame:
    """Build a forecast comparison table for selected forecast columns."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    rows = []
    for col in forecast_cols:
        if col not in df.columns:
            raise ValueError(f"Missing forecast column: {col}")
        yt, yp = _valid_pairs(df[target_col], df[col])
        rows.append(
            {
                "source": col,
                "mae": mae(yt, yp),
                "rmse": rmse(yt, yp),
                "bias": bias(yt, yp),
                "correlation": correlation(yt, yp),
                "n": int(len(yt)),
            }
        )
    return pd.DataFrame(rows)
