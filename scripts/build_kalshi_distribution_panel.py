"""Build the first Kalshi Q2 2026 market-implied GDP distribution panel.

Run from the repository root:

    python scripts/build_kalshi_distribution_panel.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.distribution import (
    build_market_distribution_panel,
    enforce_monotonic_survival_curve,
    threshold_to_bucket_probabilities,
)


RAW_PATH = RAW_DATA_DIR / "kalshi_gdp_q2_2026_thresholds.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "kalshi_q2_2026_distribution_panel.csv"
QUARTER = "2026Q2"
PLATFORM = "kalshi"


def load_threshold_snapshot(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load Kalshi thresholds and add panel metadata expected downstream."""
    df = pd.read_csv(path)
    snapshot_date = pd.to_datetime(path.stat().st_mtime, unit="s").date().isoformat()
    df = df.copy()
    df["date"] = snapshot_date
    df["quarter"] = QUARTER
    df["platform"] = PLATFORM
    return df


def validate_distribution_inputs(
    threshold_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    """Raise if the reconstructed distribution violates basic invariants."""
    if threshold_df["threshold"].notna().sum() == 0:
        raise ValueError("Threshold count is zero.")

    for _, group in clean_df.sort_values("threshold").groupby(["date", "quarter", "platform"]):
        probs = group["clean_prob_above"].to_numpy(dtype=float)
        if np.any(np.diff(probs) > 1e-12):
            raise ValueError("Clean probabilities are not monotonic non-increasing.")

    if bucket_df.empty or (bucket_df["bucket_prob"] < -1e-12).any():
        raise ValueError("Bucket probabilities are empty or negative after cleaning.")

    bucket_sums = bucket_df.groupby(["date", "quarter", "platform"])["bucket_prob"].sum()
    if not np.allclose(bucket_sums.to_numpy(dtype=float), 1.0, atol=1e-6):
        raise ValueError(f"Bucket probabilities do not sum close to 1: {bucket_sums.to_dict()}")

    required_metrics = ["market_mean_gdp", "market_median_gdp", "market_std_gdp"]
    if panel[required_metrics].isna().any().any():
        raise ValueError("Implied mean, median, or standard deviation is null.")


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return threshold, clean, bucket, and one-row distribution panel dataframes."""
    threshold_df = load_threshold_snapshot()
    clean_df = enforce_monotonic_survival_curve(threshold_df)
    bucket_df = threshold_to_bucket_probabilities(clean_df)
    panel = build_market_distribution_panel(threshold_df)
    validate_distribution_inputs(threshold_df, clean_df, bucket_df, panel)
    return threshold_df, clean_df, bucket_df, panel


def main() -> None:
    _, _, _, panel = build_panel()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
