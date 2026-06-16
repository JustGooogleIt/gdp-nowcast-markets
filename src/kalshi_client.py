"""Lightweight Kalshi market-data client for GDP research.

The initial scaffold avoids requiring credentials. Kalshi public market-data
endpoints may be usable without authentication. Authenticated requests, if
needed later, should be implemented with environment variables only and should
not include trading functionality.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import RAW_DATA_DIR


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

    def list_events(self, **params: Any) -> pd.DataFrame:
        """List Kalshi events using public market-data endpoints where possible."""
        data = self._get("/events", params=params)
        return pd.DataFrame(data.get("events", data if isinstance(data, list) else []))

    def search_gdp_events(self, query: str = "GDP", **params: Any) -> pd.DataFrame:
        """Search events likely related to GDP.

        The exact server-side search parameter should be confirmed against the
        Kalshi API docs. This method also applies a local text filter when event
        fields are available.
        """
        events = self.list_events(**params)
        if events.empty:
            return events
        text_cols = [col for col in ["ticker", "title", "subtitle", "category"] if col in events]
        if not text_cols:
            return events
        haystack = events[text_cols].fillna("").agg(" ".join, axis=1).str.lower()
        return events.loc[haystack.str.contains(query.lower(), regex=False)].copy()

    def list_markets(self, event_ticker: str, **params: Any) -> pd.DataFrame:
        """List markets for a Kalshi event ticker."""
        request_params = {"event_ticker": event_ticker, **params}
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

    def save_raw_market_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save raw market data to the raw data directory."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DATA_DIR / filename
        df.to_csv(path, index=False)
        return path
