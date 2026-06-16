"""Reconstruct GDP-implied distributions from threshold probabilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


THRESHOLD_COLUMNS = {"date", "quarter", "platform", "threshold", "prob_above"}
GROUP_COLUMNS = ["date", "quarter", "platform"]


def validate_threshold_columns(df: pd.DataFrame) -> None:
    """Validate required columns for threshold probability data."""
    missing = THRESHOLD_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required threshold columns: {sorted(missing)}")


def _normalize_probability(series: pd.Series) -> pd.Series:
    """Accept probabilities as decimals or percentages and return decimals."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().gt(1).mean() > 0.5:
        numeric = numeric / 100.0
    return numeric.clip(lower=0.0, upper=1.0)


def enforce_monotonic_survival_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Force P(GDP > threshold) to be non-increasing as thresholds rise."""
    validate_threshold_columns(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["threshold"] = pd.to_numeric(out["threshold"], errors="coerce")
    out["raw_prob_above"] = _normalize_probability(out["prob_above"])
    out = out.sort_values(GROUP_COLUMNS + ["threshold"]).reset_index(drop=True)

    def clean_group(group: pd.DataFrame) -> pd.DataFrame:
        probs = group["raw_prob_above"].to_numpy(dtype=float)
        clean = np.minimum.accumulate(probs)
        group = group.copy()
        group["clean_prob_above"] = clean
        group["monotonicity_error"] = np.maximum(clean - probs, 0.0)
        total_error = float(group["monotonicity_error"].sum())
        if total_error > 0.10:
            warnings.warn(
                "Large monotonicity correction applied to threshold probabilities",
                RuntimeWarning,
                stacklevel=2,
            )
        return group

    return out.groupby(GROUP_COLUMNS, group_keys=False).apply(clean_group)


def threshold_to_bucket_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Convert threshold survival probabilities into GDP bucket probabilities.

    For thresholds t1 < ... < tn with survival probabilities S(t), buckets are:
    (-inf, t1], (t1, t2], ..., (tn, inf). Open-ended tail midpoints are assigned
    using the adjacent threshold spacing when available.
    """
    required = THRESHOLD_COLUMNS.union({"clean_prob_above"})
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for bucket conversion: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for keys, group in df.sort_values(GROUP_COLUMNS + ["threshold"]).groupby(GROUP_COLUMNS):
        thresholds = group["threshold"].to_numpy(dtype=float)
        survival = group["clean_prob_above"].to_numpy(dtype=float)
        if len(thresholds) == 0:
            continue

        boundaries_low = np.r_[-np.inf, thresholds]
        boundaries_high = np.r_[thresholds, np.inf]
        bucket_probs = np.r_[1.0 - survival[0], survival[:-1] - survival[1:], survival[-1]]
        bucket_probs = np.maximum(bucket_probs, 0.0)
        prob_sum = bucket_probs.sum()
        if not np.isclose(prob_sum, 1.0, atol=0.05):
            warnings.warn(
                f"Bucket probabilities sum to {prob_sum:.3f} for {keys}",
                RuntimeWarning,
                stacklevel=2,
            )
        if prob_sum > 0:
            bucket_probs = bucket_probs / prob_sum

        spacings = np.diff(thresholds)
        tail_width = float(np.nanmedian(spacings)) if len(spacings) else 1.0
        tail_width = tail_width if np.isfinite(tail_width) and tail_width > 0 else 1.0

        for low, high, prob in zip(boundaries_low, boundaries_high, bucket_probs):
            if np.isneginf(low):
                mid = high - tail_width / 2.0
            elif np.isposinf(high):
                mid = low + tail_width / 2.0
            else:
                mid = (low + high) / 2.0
            rows.append(
                {
                    "date": keys[0],
                    "quarter": keys[1],
                    "platform": keys[2],
                    "bucket_low": low,
                    "bucket_high": high,
                    "bucket_mid": mid,
                    "bucket_prob": prob,
                }
            )

    return pd.DataFrame(rows)


def implied_mean_from_buckets(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """Compute probability-weighted mean GDP by date, quarter, and platform."""
    return (
        bucket_df.assign(weighted=lambda x: x["bucket_mid"] * x["bucket_prob"])
        .groupby(GROUP_COLUMNS, as_index=False)["weighted"]
        .sum()
        .rename(columns={"weighted": "market_mean_gdp"})
    )


def implied_median_from_buckets(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied median GDP with within-bucket linear interpolation."""
    rows: list[dict[str, object]] = []
    for keys, group in bucket_df.sort_values(GROUP_COLUMNS + ["bucket_mid"]).groupby(GROUP_COLUMNS):
        cumulative = group["bucket_prob"].cumsum()
        idx = cumulative.ge(0.5).idxmax()
        row = group.loc[idx]
        prev_cum = float(cumulative.loc[:idx].iloc[-2]) if cumulative.index.get_loc(idx) > 0 else 0.0
        prob = float(row["bucket_prob"])
        low = float(row["bucket_low"])
        high = float(row["bucket_high"])
        if np.isneginf(low) or np.isposinf(high) or prob <= 0:
            median = float(row["bucket_mid"])
        else:
            share = np.clip((0.5 - prev_cum) / prob, 0.0, 1.0)
            median = low + share * (high - low)
        rows.append(
            {
                "date": keys[0],
                "quarter": keys[1],
                "platform": keys[2],
                "market_median_gdp": median,
            }
        )
    return pd.DataFrame(rows)


def implied_variance_from_buckets(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """Compute probability-weighted GDP variance and standard deviation."""
    means = implied_mean_from_buckets(bucket_df)
    merged = bucket_df.merge(means, on=GROUP_COLUMNS, how="left")
    merged["weighted_sq_dev"] = (
        (merged["bucket_mid"] - merged["market_mean_gdp"]) ** 2 * merged["bucket_prob"]
    )
    out = (
        merged.groupby(GROUP_COLUMNS, as_index=False)["weighted_sq_dev"]
        .sum()
        .rename(columns={"weighted_sq_dev": "market_variance_gdp"})
    )
    out["market_std_gdp"] = np.sqrt(out["market_variance_gdp"])
    return out


def _prob_event(bucket_df: pd.DataFrame, predicate: pd.Series, name: str) -> pd.DataFrame:
    temp = bucket_df.loc[predicate].groupby(GROUP_COLUMNS, as_index=False)["bucket_prob"].sum()
    return temp.rename(columns={"bucket_prob": name})


def build_market_distribution_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build one market-implied GDP distribution row per date-quarter-platform."""
    clean = enforce_monotonic_survival_curve(df)
    buckets = threshold_to_bucket_probabilities(clean)
    mean = implied_mean_from_buckets(buckets)
    median = implied_median_from_buckets(buckets)
    variance = implied_variance_from_buckets(buckets)

    below_0 = _prob_event(buckets, buckets["bucket_high"].le(0), "prob_gdp_below_0")
    above_2 = _prob_event(buckets, buckets["bucket_low"].ge(2), "prob_gdp_above_2")
    above_3 = _prob_event(buckets, buckets["bucket_low"].ge(3), "prob_gdp_above_3")
    monotonicity = (
        clean.groupby(GROUP_COLUMNS, as_index=False)["monotonicity_error"]
        .sum()
        .rename(columns={"monotonicity_error": "monotonicity_error"})
    )

    panel = mean.merge(median, on=GROUP_COLUMNS, how="outer")
    panel = panel.merge(variance[GROUP_COLUMNS + ["market_std_gdp"]], on=GROUP_COLUMNS, how="outer")
    for event_df in [below_0, above_2, above_3, monotonicity]:
        panel = panel.merge(event_df, on=GROUP_COLUMNS, how="left")

    for col in ["prob_gdp_below_0", "prob_gdp_above_2", "prob_gdp_above_3", "monotonicity_error"]:
        panel[col] = panel[col].fillna(0.0)
    return panel[
        [
            "date",
            "quarter",
            "platform",
            "market_mean_gdp",
            "market_median_gdp",
            "market_std_gdp",
            "prob_gdp_below_0",
            "prob_gdp_above_2",
            "prob_gdp_above_3",
            "monotonicity_error",
        ]
    ]
