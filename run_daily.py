"""
run_daily.py

End-to-end daily NBA Pro-Lite pipeline.

Steps:
1) Fetch today's NBA games from API-Sports (nba_ingest).
2) Build a games DataFrame (game_id, game_date, home_team, away_team).
3) Use the trained balldontlie-based model (predict_games) to generate:
       - home_win_prob
       - away_win_prob
       - fair_spread
       - fair_total
   and write outputs/predictions_<YYYY-MM-DD>.csv
4) Apply market ensemble (if odds dispersion file available) to produce
       outputs/predictions_<YYYY-MM-DD>_market.csv
5) Run edge picker to generate:
       - outputs/picks_<YYYY-MM-DD>.csv
       - picks_report.html

Environment:
- RAPIDAPI_KEY   : your API-Sports key (for nba_ingest)
- ODDS_API_KEY   : your The Odds API key (used elsewhere in the pipeline)
- RUN_DATE       : optional override for the run date (YYYY-MM-DD). If not set,
                   we use today's date in UTC/local.

Usage (from repo root):
    PYTHONPATH=. python run_daily.py
"""

from __future__ import annotations

import os
from datetime import date
from typing import List, Dict, Any

import pandas as pd

from src.ingest.nba_ingest import get_nba_games_by_date
from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import main as run_edge_picker


def _get_run_date() -> str:
    """Return the run date as YYYY-MM-DD, honoring RUN_DATE if set."""
    return os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")


def fetch_today_games(run_date: str) -> pd.DataFrame:
    """
    Fetch today's NBA games from API-Sports via nba_ingest.get_nba_games_by_date.

    Returns a DataFrame with:
        game_id, game_date, home_team, away_team
    """
    raw = get_nba_games_by_date(run_date)
    games: List[Dict[str, Any]] = raw.get("response", [])

    rows: List[Dict[str, Any]] = []
    for g in games:
        gid = g.get("id")
        teams = g.get("teams", {}) or {}
        home = teams.get("home", {}) or {}
        away = teams.get("visitors", {}) or {}

        home_name = home.get("name")
        away_name = away.get("name")

        if not home_name or not away_name:
            continue

        rows.append(
            {
                "game_id": gid,
                "game_date": run_date,
                "home_team": home_name,
                "away_team": away_name,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[run_daily] No NBA games found for {run_date}.")
    else:
        print(f"[run_daily] Fetched {len(df)} games for {run_date}.")
    return df


def build_model_predictions(games_df: pd.DataFrame, run_date: str) -> str:
    """
    Use the trained models to generate predictions for today's games and
    write outputs/predictions_<run_date>.csv.

    Returns the path to the predictions CSV.
    """
    if games_df.empty:
        raise RuntimeError("[run_daily] No games DataFrame provided to build_model_predictions.")

    preds = predict_games(games_df)

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/predictions_{run_date}.csv"
    preds.to_csv(out_path, index=False)
    print(f"[run_daily] Wrote base predictions to {out_path} ({len(preds)} rows)")
    return out_path


def apply_market_adjustment(preds_path: str, run_date: str) -> str:
    """
    Apply market-aware ensemble adjustment if odds dispersion is available.

    Reads:
        - preds_path (base predictions)
        - outputs/odds_dispersion_latest.csv (if exists)

    Writes:
        - outputs/predictions_<run_date>_market.csv

    Returns the path to the market-adjusted predictions CSV.
    """
    preds = pd.read_csv(preds_path)

    odds_path = "outputs/odds_dispersion_latest.csv"
    if os.path.exists(odds_path):
        print(f"[run_daily] Found {odds_path}; applying market ensemble.")
        odds = pd.read_csv(odds_path)
    else:
        print("[run_daily] No odds_dispersion_latest.csv found; applying model-only ensemble.")
        odds = None

    out = apply_market_ensemble(preds, odds)
    out_path = f"outputs/predictions_{run_date}_market.csv"
    out.to_csv(out_path, index=False)
    print(f"[run_daily] Wrote market-adjusted predictions to {out_path} ({len(out)} rows)")
    return out_path


def run_picks_pipeline(run_date: str) -> None:
    """
    Run the edge picker to generate picks CSV and HTML report.

    edge_picker.load_predictions_for_today() will automatically prefer the
    market-adjusted file outputs/predictions_<run_date>_market.csv if it exists.
    """
    print("[run_daily] Running edge picker to generate pick sheet and HTML report...")
    # edge_picker.main() handles:
    #   - loading today's predictions
    #   - reading snapshot/dispersion if available
    #   - writing outputs/picks_<run_date>.csv
    #   - writing picks_report.html
    run_edge_picker()
    print("[run_daily] Edge picker completed.")


def main():
    run_date = _get_run_date()
    print(f"[run_daily] Starting daily pipeline for {run_date}")

    # 1) Fetch today's games from API-Sports
    games_df = fetch_today_games(run_date)
    if games_df.empty:
        print("[run_daily] No games today; exiting gracefully.")
        return

    # 2) Build model predictions (using trained balldontlie-based models)
    base_preds_path = build_model_predictions(games_df, run_date)

    # 3) Apply market ensemble adjustment (if odds dispersion is present)
    market_preds_path = apply_market_adjustment(base_preds_path, run_date)

    # 4) Run edge picker to produce picks CSV + HTML
    run_picks_pipeline(run_date)

    print("[run_daily] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
