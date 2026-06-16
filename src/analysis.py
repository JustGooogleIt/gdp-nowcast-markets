"""Structured analysis helpers for forecast comparisons and disagreement tests."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from .metrics import forecast_comparison_table


def build_hybrid_forecast(panel_df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Build a convex-combination institutional and market forecast."""
    required = {"institutional_mean", "market_mean_gdp"}
    missing = required.difference(panel_df.columns)
    if missing:
        raise ValueError(f"Missing required hybrid forecast columns: {sorted(missing)}")
    out = panel_df.copy()
    out["hybrid_forecast"] = alpha * out["institutional_mean"] + (1 - alpha) * out["market_mean_gdp"]
    if "advance_gdp" in out.columns:
        out["hybrid_error"] = out["hybrid_forecast"] - out["advance_gdp"]
        out["abs_hybrid_error"] = out["hybrid_error"].abs()
        out["squared_hybrid_error"] = out["hybrid_error"] ** 2
    return out


def forecast_horse_race(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compare institutional, platform market, market-average, and hybrid forecasts."""
    if "advance_gdp" not in panel_df.columns:
        raise ValueError("forecast_horse_race requires advance_gdp")

    working = panel_df.copy()
    forecast_cols = []
    if "institutional_mean" in working.columns:
        forecast_cols.append("institutional_mean")
    if "market_mean_gdp" in working.columns:
        forecast_cols.append("market_mean_gdp")
    if {"institutional_mean", "market_mean_gdp"}.issubset(working.columns):
        working = build_hybrid_forecast(working)
        forecast_cols.append("hybrid_forecast")

    if "platform" in working.columns and "market_mean_gdp" in working.columns:
        market_average = (
            working.groupby(["date", "quarter"], as_index=False)["market_mean_gdp"]
            .mean()
            .rename(columns={"market_mean_gdp": "market_average_gdp"})
        )
        working = working.merge(market_average, on=["date", "quarter"], how="left")
        forecast_cols.append("market_average_gdp")
        wide = working.pivot_table(
            index=["date", "quarter"],
            columns="platform",
            values="market_mean_gdp",
            aggfunc="mean",
        ).reset_index()
        wide.columns = [
            col if not isinstance(col, tuple) else "_".join(str(x) for x in col if x)
            for col in wide.columns
        ]
        platform_cols = [col for col in wide.columns if col not in {"date", "quarter"}]
        realized_cols = ["date", "quarter", "advance_gdp"]
        realized = working[realized_cols].drop_duplicates(["date", "quarter"])
        wide = wide.merge(realized, on=["date", "quarter"], how="left")
        platform_table = forecast_comparison_table(wide, "advance_gdp", platform_cols)
        pooled_table = forecast_comparison_table(working, "advance_gdp", forecast_cols)
        return pd.concat([pooled_table, platform_table], ignore_index=True)

    return forecast_comparison_table(working, "advance_gdp", forecast_cols)


def disagreement_predicts_forecast_error(panel_df: pd.DataFrame):
    """Regress institutional forecast error on market-institutional disagreement."""
    required = {"institutional_error", "market_minus_institutional"}
    missing = required.difference(panel_df.columns)
    if missing:
        raise ValueError(f"Missing required regression columns: {sorted(missing)}")
    data = panel_df[list(required)].dropna()
    y = data["institutional_error"]
    x = sm.add_constant(data["market_minus_institutional"])
    return sm.OLS(y, x).fit()


def disagreement_predicts_nowcast_revision(panel_df: pd.DataFrame):
    """Regress future institutional nowcast revision on current disagreement."""
    required = {"date", "quarter", "institutional_mean", "market_minus_institutional"}
    missing = required.difference(panel_df.columns)
    if missing:
        raise ValueError(f"Missing required revision regression columns: {sorted(missing)}")

    data = panel_df.sort_values(["quarter", "date"]).copy()
    data["future_institutional_nowcast"] = data.groupby("quarter")["institutional_mean"].shift(-1)
    data["future_nowcast_revision"] = (
        data["future_institutional_nowcast"] - data["institutional_mean"]
    )
    reg = data[["future_nowcast_revision", "market_minus_institutional"]].dropna()
    y = reg["future_nowcast_revision"]
    x = sm.add_constant(reg["market_minus_institutional"])
    return sm.OLS(y, x).fit()
