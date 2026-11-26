# TheLotto Client App — v4

## Overview
Client-facing Streamlit prediction app. Runs the same walk-forward prediction pipeline as the main system.

## What Changed in v4
1. **Fixed prediction pipeline** — `newx_strategy_results.json` and `market_assign.json` are now synced with the latest main-system data (extracted from the Nov 2026 HTML dashboard). This was the primary cause of prediction differences.
2. **SupremeNight override** — now correctly forces `Mean-Revert-30` strategy (matches daily_runner.py).
3. **Model B blending** — 60/40 blend code now matches main system exactly.
4. **Min rows fix** — threshold changed from 15 to 10 (matches main system).
5. **Pandas 2.1 fix** — `.ffill().bfill()` instead of deprecated `fillna(method='ffill')`.

## Deployment (Streamlit Community Cloud)
1. Push this folder to your private GitHub repo.
2. Connect the repo to Streamlit Community Cloud.
3. Set `client_app.py` as the main file.
4. No secrets or environment variables needed.

## Data Updates
- `data/bt_data.json` — Update weekly by exporting from your main system.
- `data/market_assign.json` — Update when strategy assignments change.
- `data/newx_strategy_results.json` — Update when NX model is re-optimized.

## Security
Strategy names, methodology, and NX experiment details are NOT exposed in the UI.
