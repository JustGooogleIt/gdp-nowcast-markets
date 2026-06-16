"""Pull Kalshi Q2 2026 GDP threshold market data.

Run from the repository root:

    python scripts/pull_kalshi_gdp_q2_2026.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kalshi_client import KalshiClient


def main() -> None:
    client = KalshiClient()
    path = client.save_gdp_q2_2026_thresholds()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
