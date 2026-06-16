"""Lightweight Polymarket data client for GDP and macro market research."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import RAW_DATA_DIR


class PolymarketClient:
    """Client skeleton for Polymarket public and optional API-key data calls."""

    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
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

    def search_gdp_markets(self, query: str = "GDP", **params: Any) -> pd.DataFrame:
        """Search for GDP-related markets.

        Endpoint and query parameter details should be confirmed against current
        Polymarket documentation before production use.
        """
        data = self._get("/markets", params={"search": query, **params})
        return pd.DataFrame(data if isinstance(data, list) else data.get("markets", []))

    def search_macro_markets(self, query: str = "macro", **params: Any) -> pd.DataFrame:
        """Search for broader macro markets potentially related to GDP expectations."""
        data = self._get("/markets", params={"search": query, **params})
        return pd.DataFrame(data if isinstance(data, list) else data.get("markets", []))

    def fetch_market_history(self, market_id: str, **params: Any) -> pd.DataFrame:
        """Fetch market history for a Polymarket market id.

        Exact history endpoint details are placeholders pending documentation
        review.
        """
        data = self._get(f"/markets/{market_id}/history", params=params)
        rows = data if isinstance(data, list) else data.get("history", [])
        return pd.DataFrame(rows)

    def save_raw_market_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save raw market data to the raw data directory."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DATA_DIR / filename
        df.to_csv(path, index=False)
        return path
