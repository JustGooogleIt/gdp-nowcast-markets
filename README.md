# GDP Nowcast Markets

This project compares institutional and market-based forecasts of U.S. quarterly real GDP growth. Institutional nowcasts, such as the Atlanta Fed GDPNow and New York Fed Staff Nowcast, translate released macroeconomic source data into estimates of current-quarter growth. Prediction markets, such as Kalshi and Polymarket, aggregate traded beliefs that may also reflect soft information, interpretation risk, macro sentiment, recession concerns, liquidity conditions, and uncertainty about future revisions. The project evaluates which forecasting system is more accurate and whether disagreement between the two systems contains incremental information.

## Research Question

Do prediction-market-implied GDP expectations from Kalshi and Polymarket improve on institutional quarterly GDP nowcasts, and does the discrepancy between market-based and institutional forecasts contain useful information about future nowcast revisions, BEA GDP releases, or macro uncertainty?

## Why Quarterly GDP Nowcasting?

Quarterly real GDP growth is a focal summary statistic for U.S. macroeconomic conditions, but it is released with a lag and revised over time. During a quarter, analysts, policymakers, and investors use nowcasts to infer growth before the Bureau of Economic Analysis publishes official estimates. That makes GDP nowcasting a useful setting for studying how structured statistical systems and traded market beliefs process information in real time.

## Institutional vs. Market-Based Forecasts

Institutional nowcasts are usually model-driven estimates based on incoming macroeconomic source data. They are transparent about update timing and often tied to official release calendars.

Market-based nowcasts are inferred from traded contracts. For GDP-linked prediction markets, prices can be transformed into threshold probabilities and reconstructed into an implied distribution over quarterly GDP growth. These probabilities may embed information outside standard release calendars, including sentiment, hedging demand, liquidity, and disagreement among informed participants.

## Why Disagreement May Matter

The gap between market-implied GDP expectations and institutional nowcasts may contain incremental information. A persistent market premium or discount relative to institutional nowcasts could predict:

- future institutional nowcast revisions
- BEA advance, second, third, or latest GDP releases
- forecast errors in either system
- macro uncertainty or stress

This repository is designed to test those possibilities without inventing data or creating fake outputs.

## Expected Data Sources

- Institutional GDP nowcasts, such as Atlanta Fed GDPNow and New York Fed Staff Nowcast
- Kalshi GDP threshold market probabilities
- Polymarket GDP or macro market probabilities
- BEA quarterly GDP release vintages
- Optional macro uncertainty controls in later extensions

## Repository Structure

```text
gdp-nowcast-markets/
  data/
    raw/
    processed/
    external/
  notebooks/
    01_kalshi_gdp_pull.ipynb
    02_distribution_reconstruction.ipynb
    03_forecast_comparison.ipynb
  outputs/
    tables/
    figures/
  src/
    config.py
    kalshi_client.py
    polymarket_client.py
    distribution.py
    nowcasts.py
    merge_panel.py
    metrics.py
    analysis.py
```

## Setup

```bash
cd gdp-nowcast-markets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The initial scaffold does not require a Kalshi API key. Public market-data endpoints should be used without authentication where possible. Authenticated access, if needed later, should use environment variables only:

- `KALSHI_API_KEY_ID`
- `KALSHI_PRIVATE_KEY_PATH`
- `POLYMARKET_API_KEY`

Do not commit a real `.env` file or private key material.

## Expected Pipeline

1. Pull or manually stage raw prediction-market data in `data/raw/`.
2. Convert GDP threshold probabilities into market-implied distributions.
3. Load institutional nowcasts and aggregate across sources.
4. Load BEA quarterly GDP releases.
5. Merge a daily date-quarter-platform panel.
6. Compute disagreement variables and forecast errors.
7. Compare institutional, market-implied, and hybrid forecasts.
8. Test whether disagreement predicts revisions, realized GDP, or forecast errors.

## Current Status

This is an initial research scaffold. It includes project structure, configuration, client skeletons, validation utilities, distribution reconstruction logic, panel merging helpers, metrics, and analysis placeholders. It does not include raw data, generated tables, generated figures, API keys, or fake outputs.
