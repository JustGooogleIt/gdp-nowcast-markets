"""Load and aggregate institutional nowcasts and BEA GDP releases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NOWCAST_COLUMNS = {"date", "quarter", "source", "nowcast"}
BEA_COLUMNS = {
    "quarter",
    "advance_release_date",
    "advance_gdp",
    "second_release_date",
    "second_gdp",
    "third_release_date",
    "third_gdp",
    "latest_gdp",
}


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required {label} columns: {sorted(missing)}")


def load_institutional_nowcasts(path: str | Path) -> pd.DataFrame:
    """Load institutional nowcasts from CSV and validate core columns."""
    df = pd.read_csv(path)
    _require_columns(df, NOWCAST_COLUMNS, "institutional nowcast")
    df["date"] = pd.to_datetime(df["date"])
    df["nowcast"] = pd.to_numeric(df["nowcast"], errors="coerce")
    return df


def aggregate_institutional_nowcasts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate institutional nowcasts to one row per date and quarter."""
    _require_columns(df, NOWCAST_COLUMNS, "institutional nowcast")
    out = (
        df.groupby(["date", "quarter"])["nowcast"]
        .agg(
            institutional_mean="mean",
            institutional_median="median",
            institutional_std="std",
            num_institutional_sources="count",
        )
        .reset_index()
    )
    out["institutional_std"] = out["institutional_std"].fillna(0.0)
    return out


def load_bea_gdp_releases(path: str | Path) -> pd.DataFrame:
    """Load BEA GDP release vintages from CSV."""
    df = pd.read_csv(path)
    _require_columns(df, BEA_COLUMNS, "BEA GDP release")
    for col in ["advance_release_date", "second_release_date", "third_release_date"]:
        df[col] = pd.to_datetime(df[col])
    for col in ["advance_gdp", "second_gdp", "third_gdp", "latest_gdp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
