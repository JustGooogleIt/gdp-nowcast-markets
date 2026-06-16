"""Lightweight Kalshi market-data client for GDP research.

The initial scaffold avoids requiring credentials. Kalshi public market-data
endpoints may be usable without authentication. Authenticated requests, if
needed later, should be implemented with environment variables only and should
not include trading functionality.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import RAW_DATA_DIR


GDP_Q2_2026_EVENT_TITLE = "US GDP growth in Q2 2026?"
GDP_Q2_2026_OUTPUT = "kalshi_gdp_q2_2026_thresholds.csv"


class KalshiClient:
    """Client skeleton for Kalshi public and optional authenticated data calls."""

    def __init__(
        self,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        api_key_id: str | None = None,
        private_key_path: str | Path | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID")
        self.private_key_path = Path(
            private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
        )
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def has_auth_config(self) -> bool:
        """Return True when optional Kalshi auth environment variables are set."""
        return bool(self.api_key_id and str(self.private_key_path))

    def _headers(self) -> dict[str, str]:
        """Return request headers.

        Authenticated signing is intentionally left as a future implementation
        detail pending confirmation against the current Kalshi API docs.
        """
        headers = {"accept": "application/json"}
        if self.has_auth_config:
            headers["KALSHI-ACCESS-KEY"] = str(self.api_key_id)
        return headers

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a GET request and return decoded JSON."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(
            url,
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _paginate(self, endpoint: str, rows_key: str, **params: Any) -> list[dict[str, Any]]:
        """Collect all pages from a Kalshi list endpoint."""
        rows: list[dict[str, Any]] = []
        request_params = dict(params)
        while True:
            data = self._get(endpoint, params=request_params)
            page_rows = data.get(rows_key, data if isinstance(data, list) else [])
            rows.extend(page_rows)
            cursor = data.get("cursor") if isinstance(data, dict) else None
            if not cursor:
                return rows
            request_params["cursor"] = cursor

    def list_events(self, paginate: bool = True, **params: Any) -> pd.DataFrame:
        """List Kalshi events using public market-data endpoints where possible."""
        if paginate:
            return pd.DataFrame(self._paginate("/events", "events", **params))
        data = self._get("/events", params=params)
        return pd.DataFrame(data.get("events", data if isinstance(data, list) else []))

    def search_gdp_events(self, query: str = "GDP", **params: Any) -> pd.DataFrame:
        """Search events likely related to GDP.

        Kalshi's public event list supports filtering by series ticker. The GDP
        growth series is currently KXGDP; this method still applies a local text
        filter so it remains useful if callers pass broader parameters.
        """
        params.setdefault("series_ticker", "KXGDP")
        params.setdefault("limit", 200)
        events = self.list_events(**params)
        if events.empty:
            return events
        text_cols = [col for col in ["ticker", "title", "subtitle", "category"] if col in events]
        if not text_cols:
            return events
        haystack = events[text_cols].fillna("").agg(" ".join, axis=1).str.lower()
        return events.loc[haystack.str.contains(query.lower(), regex=False)].copy()

    def find_event_by_title(
        self,
        title: str,
        *,
        series_ticker: str | None = "KXGDP",
        **params: Any,
    ) -> dict[str, Any]:
        """Find one Kalshi event by exact title, preferring active/open events."""
        if series_ticker:
            params.setdefault("series_ticker", series_ticker)
        params.setdefault("limit", 200)
        events = self.list_events(**params)
        if events.empty or "title" not in events:
            raise ValueError(f"No Kalshi events found while looking for {title!r}.")

        matches = events.loc[events["title"].fillna("").str.casefold() == title.casefold()]
        if matches.empty:
            available = events.get("title", pd.Series(dtype=str)).dropna().tolist()
            raise ValueError(f"Could not find Kalshi event {title!r}. Available titles: {available}")

        for status_col in ["status", "event_status"]:
            if status_col in matches:
                active = matches.loc[
                    matches[status_col].fillna("").str.casefold().isin(["active", "open"])
                ]
                if not active.empty:
                    return active.iloc[0].to_dict()
        return matches.iloc[0].to_dict()

    def list_markets(self, event_ticker: str, paginate: bool = True, **params: Any) -> pd.DataFrame:
        """List markets for a Kalshi event ticker."""
        request_params = {"event_ticker": event_ticker, **params}
        request_params.setdefault("limit", 200)
        if paginate:
            return pd.DataFrame(self._paginate("/markets", "markets", **request_params))
        data = self._get("/markets", params=request_params)
        return pd.DataFrame(data.get("markets", data if isinstance(data, list) else []))

    def fetch_market(self, market_ticker: str) -> dict[str, Any]:
        """Fetch metadata for one market ticker."""
        return self._get(f"/markets/{market_ticker}")

    def fetch_market_history(self, market_ticker: str, **params: Any) -> pd.DataFrame:
        """Fetch historical market data.

        Endpoint details and returned fields should be confirmed before use.
        """
        data = self._get(f"/markets/{market_ticker}/history", params=params)
        rows = data.get("history", data.get("candlesticks", data if isinstance(data, list) else []))
        return pd.DataFrame(rows)

    @staticmethod
    def parse_threshold(contract: pd.Series | dict[str, Any]) -> float | None:
        """Parse threshold text such as 'Above 2.5%' into a numeric value."""
        fields = ["yes_sub_title", "subtitle", "title", "ticker", "rules_primary"]
        row = dict(contract)
        for field in fields:
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            text = str(value)
            above_match = re.search(r"\b(?:above|more than|greater than)\s*(-?\d+(?:\.\d+)?)\s*%", text, re.I)
            if above_match:
                return float(above_match.group(1))
            ticker_match = re.search(r"-T(-?\d+(?:\.\d+)?)$", text)
            if ticker_match:
                return float(ticker_match.group(1))
        floor_strike = row.get("floor_strike")
        if floor_strike is not None and not pd.isna(floor_strike):
            return float(floor_strike)
        return None

    @staticmethod
    def _price_to_probability(value: Any) -> float | None:
        """Normalize Kalshi price fields into 0-1 probabilities."""
        if value is None or pd.isna(value):
            return None
        price = float(value)
        if price > 1:
            return price / 100
        return price

    @classmethod
    def _first_probability(cls, row: pd.Series, fields: list[str]) -> float | None:
        for field in fields:
            if field in row and not pd.isna(row[field]):
                return cls._price_to_probability(row[field])
        return None

    @classmethod
    def add_threshold_probabilities(cls, markets: pd.DataFrame) -> pd.DataFrame:
        """Add threshold, yes_mid, and prob_above columns to GDP markets."""
        if markets.empty:
            return markets.copy()

        df = markets.copy()
        df["threshold"] = df.apply(cls.parse_threshold, axis=1)

        yes_bid = df.apply(
            lambda row: cls._first_probability(row, ["yes_bid_dollars", "yes_bid"]),
            axis=1,
        )
        yes_ask = df.apply(
            lambda row: cls._first_probability(row, ["yes_ask_dollars", "yes_ask"]),
            axis=1,
        )
        last_price = df.apply(
            lambda row: cls._first_probability(row, ["last_price_dollars", "last_price"]),
            axis=1,
        )

        df["yes_bid_prob"] = yes_bid
        df["yes_ask_prob"] = yes_ask
        df["last_price_prob"] = last_price
        df["yes_mid"] = (yes_bid + yes_ask) / 2
        df["prob_above"] = df["yes_mid"].combine_first(df["last_price_prob"])
        return df.sort_values("threshold", na_position="last").reset_index(drop=True)

    def pull_gdp_q2_2026_thresholds(self) -> pd.DataFrame:
        """Pull the active Kalshi Q2 2026 GDP threshold contracts.

        TODO: Add historical time-series pulls once the correct public
        candlestick/history endpoint and interval semantics are confirmed.
        TODO: Store market snapshots with retrieval timestamps when building a
        daily panel from repeated pulls.
        """
        event = self.find_event_by_title(GDP_Q2_2026_EVENT_TITLE, series_ticker="KXGDP")
        event_ticker = event["event_ticker"]
        markets = self.list_markets(event_ticker=event_ticker)
        if markets.empty:
            raise ValueError(f"No markets returned for Kalshi event {event_ticker}.")

        thresholds = self.add_threshold_probabilities(markets)
        thresholds.insert(0, "source_event_title", event["title"])
        thresholds.insert(0, "source_event_ticker", event_ticker)
        return thresholds

    def save_gdp_q2_2026_thresholds(
        self,
        filename: str = GDP_Q2_2026_OUTPUT,
    ) -> Path:
        """Pull and save the public Kalshi Q2 2026 GDP threshold snapshot."""
        df = self.pull_gdp_q2_2026_thresholds()
        return self.save_raw_market_data(df, filename)

    def save_raw_market_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save raw market data to the raw data directory."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DATA_DIR / filename
        df.to_csv(path, index=False)
        return path
