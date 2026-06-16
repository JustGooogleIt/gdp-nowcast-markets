"""Merge institutional, market-implied, and realized GDP panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def merge_forecast_panel(
    institutional_df: pd.DataFrame,
    market_df: pd.DataFrame,
    bea_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a daily date-quarter-platform forecast panel."""
    for label, df, cols in [
        ("institutional", institutional_df, {"date", "quarter"}),
        ("market", market_df, {"date", "quarter", "platform"}),
        ("BEA", bea_df, {"quarter"}),
    ]:
        missing = cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required {label} columns: {sorted(missing)}")

    panel = market_df.merge(institutional_df, on=["date", "quarter"], how="left")
    panel = panel.merge(bea_df, on="quarter", how="left")
    return panel


def add_disagreement_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-institutional disagreement variables in long or wide format."""
    out = df.copy()
    if {"market_mean_gdp", "institutional_mean"}.issubset(out.columns):
        out["market_minus_institutional"] = out["market_mean_gdp"] - out["institutional_mean"]

    if "platform" in out.columns and "market_minus_institutional" in out.columns:
        platform_lower = out["platform"].astype(str).str.lower()
        out["kalshi_minus_institutional"] = np.where(
            platform_lower.eq("kalshi"),
            out["market_minus_institutional"],
            np.nan,
        )
        out["polymarket_minus_institutional"] = np.where(
            platform_lower.eq("polymarket"),
            out["market_minus_institutional"],
            np.nan,
        )

    wide_pairs = [
        ("kalshi_market_mean_gdp", "kalshi_minus_institutional"),
        ("polymarket_market_mean_gdp", "polymarket_minus_institutional"),
    ]
    if "institutional_mean" in out.columns:
        for source_col, disagreement_col in wide_pairs:
            if source_col in out.columns:
                out[disagreement_col] = out[source_col] - out["institutional_mean"]
    return out


def add_forecast_errors(df: pd.DataFrame, target_col: str = "advance_gdp") -> pd.DataFrame:
    """Add institutional and market forecast error columns."""
    required = {target_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing target column for forecast errors: {sorted(missing)}")

    out = df.copy()
    if "institutional_mean" in out.columns:
        out["institutional_error"] = out["institutional_mean"] - out[target_col]
        out["abs_institutional_error"] = out["institutional_error"].abs()
        out["squared_institutional_error"] = out["institutional_error"] ** 2
    if "market_mean_gdp" in out.columns:
        out["market_error"] = out["market_mean_gdp"] - out[target_col]
        out["abs_market_error"] = out["market_error"].abs()
        out["squared_market_error"] = out["market_error"] ** 2
    return out
