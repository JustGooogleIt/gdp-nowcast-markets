"""Build the Q2 2026 Kalshi vs institutional forecast comparison panel.

Run from the repository root:

    python scripts/build_q2_2026_comparison_panel.py
"""

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from src.nowcasts import aggregate_institutional_nowcasts, load_institutional_nowcasts


KALSHI_PATH = PROCESSED_DATA_DIR / "kalshi_q2_2026_distribution_panel.csv"
NOWCAST_PATH = EXTERNAL_DATA_DIR / "institutional_nowcasts.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "q2_2026_forecast_comparison_panel.csv"
COMPARISON_COLUMNS = [
    "date",
    "quarter",
    "kalshi_mean_gdp",
    "kalshi_median_gdp",
    "kalshi_std_gdp",
    "institutional_mean",
    "market_minus_institutional",
]


def require_input_file(path: Path) -> None:
    """Validate that a required real input file exists and is non-empty."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required input file is empty: {path}")


def load_kalshi_distribution(path: Path = KALSHI_PATH) -> pd.DataFrame:
    """Load Kalshi distribution panel and standardize comparison columns."""
    require_input_file(path)
    df = pd.read_csv(path)
    required = {"date", "quarter", "market_mean_gdp", "market_median_gdp", "market_std_gdp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required Kalshi distribution columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Kalshi distribution dates did not parse correctly.")

    return df.rename(
        columns={
            "market_mean_gdp": "kalshi_mean_gdp",
            "market_median_gdp": "kalshi_median_gdp",
            "market_std_gdp": "kalshi_std_gdp",
        }
    )


def load_institutional_panel(path: Path = NOWCAST_PATH) -> pd.DataFrame:
    """Load and aggregate institutional nowcasts by date and quarter."""
    require_input_file(path)
    nowcasts = load_institutional_nowcasts(path)
    if nowcasts["date"].isna().any():
        raise ValueError("Institutional nowcast dates did not parse correctly.")
    panel = aggregate_institutional_nowcasts(nowcasts)
    if panel["institutional_mean"].isna().any():
        raise ValueError("institutional_mean is null after aggregation.")
    return panel


def build_comparison_panel() -> pd.DataFrame:
    """Merge Kalshi and institutional forecasts and compute disagreement."""
    kalshi = load_kalshi_distribution()
    institutional = load_institutional_panel()

    kalshi_quarters = set(kalshi["quarter"].dropna().astype(str))
    institutional_quarters = set(institutional["quarter"].dropna().astype(str))
    if not kalshi_quarters.intersection(institutional_quarters):
        raise ValueError(
            "Quarter values do not match between Kalshi and institutional inputs: "
            f"kalshi={sorted(kalshi_quarters)}, institutional={sorted(institutional_quarters)}"
        )

    panel = kalshi.merge(institutional, on=["date", "quarter"], how="inner")
    if panel.empty:
        raise ValueError("No matching date-quarter rows after merging Kalshi and institutional panels.")

    panel["market_minus_institutional"] = panel["kalshi_mean_gdp"] - panel["institutional_mean"]
    if panel["market_minus_institutional"].isna().any():
        raise ValueError("market_minus_institutional was not computed.")

    return panel


def comparison_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the compact table requested for inspection."""
    return panel[COMPARISON_COLUMNS].sort_values(["date", "quarter"]).reset_index(drop=True)


def main() -> None:
    panel = build_comparison_panel()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    print(comparison_table(panel).to_string(index=False))


if __name__ == "__main__":
    main()
