"""
run_daily.py

End-to-end daily NBA Pro-Lite pipeline.

Steps:
1) Try to fetch today's NBA games from API-Sports (nba_ingest).
2) If none are found, fall back to using odds_dispersion_latest.csv
   to infer the slate (home_team, away_team, game_date).
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
6) Write a simple run_summary.md.

Environment:
- RAPIDAPI_KEY   : API-Sports key (for nba_ingest)
- ODDS_API_KEY   : The Odds API key (used in odds_snapshots workflow)
- RUN_DATE       : optional override (YYYY-MM-DD); default is today's date
"""

from __future__ import annotations

import os
from datetime import date
from typing import List, Dict, Any

import pandas as pd

from src.ingest.nba_ingest import get_nba_games_by_date
from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import main as run_edge_picker, _merge_key as _ep_merge_key


def _get_run_date() -> str:
    return os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")


def fetch_today_games_from_api(run_date: str) -> pd.DataFrame:
    """
    Fetch today's NBA games from API-Sports via nba_ingest.get_nba_games_by_date.

    Returns DataFrame with:
        game_id, game_date, home_team, away_team
    (May be empty if API-Sports has no games for that date.)
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
        print(f"[run_daily] API-Sports returned no games for {run_date}.")
    else:
        print(f"[run_daily] Fetched {len(df)} games from API-Sports for {run_date}.")
    return df


def fetch_today_games_from_dispersion(run_date: str) -> pd.DataFrame:
    """
    Fallback: infer today's slate from outputs/odds_dispersion_latest.csv.

    Returns DataFrame with:
        game_id, game_date, home_team, away_team
    where game_id is a synthetic ID equal to merge_key.
    """
    odds_path = "outputs/odds_dispersion_latest.csv"
    if not os.path.exists(odds_path):
        print("[run_daily] No odds_dispersion_latest.csv found; cannot infer slate.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    disp = pd.read_csv(odds_path)
    if disp.empty:
        print("[run_daily] odds_dispersion_latest.csv is empty; cannot infer slate.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    if "game_date" not in disp.columns:
        print("[run_daily] dispersion missing game_date; cannot infer slate.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    day = disp[disp["game_date"] == run_date].copy()
    if day.empty:
        print(f"[run_daily] dispersion has no rows for {run_date}; cannot infer slate.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    # One row per unique matchup on that date
    day = day.sort_values("merge_key").drop_duplicates("merge_key")

    games_df = day[["home_team", "away_team", "game_date", "merge_key"]].copy()
    games_df.rename(columns={"merge_key": "game_id"}, inplace=True)
    print(f"[run_daily] Inferred {len(games_df)} games from dispersion for {run_date}.")
    return games_df


def build_model_predictions(games_df: pd.DataFrame, run_date: str) -> str:
    """
    Use the trained models to generate predictions for today's games and
    write outputs/predictions_<run_date>.csv.

    Adds a merge_key column so market_ensemble and edge_picker can join
    on odds data reliably.

    Returns the path to the predictions CSV.
    """
    if games_df.empty:
        raise RuntimeError("[run_daily] No games DataFrame provided to build_model_predictions.")

    preds = predict_games(games_df)

    # Ensure game_date is present
    if "game_date" not in preds.columns:
        preds["game_date"] = run_date

    # Add merge_key using the same logic as edge_picker
    preds["merge_key"] = preds.apply(
        lambda r: _ep_merge_key(r["home_team"], r["away_team"], r["game_date"]), axis=1
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/predictions_{run_date}.csv"
    preds.to_csv(out_path, index=False)
    print(f"[run_daily] Wrote base predictions to {out_path} ({len(preds)} rows)")
    return out_path


def apply_market_adjustment(preds_path: str, run_date: str) -> str:
    """
    Apply market-aware ensemble adjustment if odds dispersion is available.
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
    run_edge_picker()
    print("[run_daily] Edge picker completed.")


def write_run_summary(run_date: str, n_games: int) -> None:
    text = f"# NBA daily run\n\nDate: {run_date}\nGames predicted: {n_games}\n"
    with open("run_summary.md", "w") as f:
        f.write(text)
    print("[run_daily] Wrote run_summary.md")


def main():
    run_date = _get_run_date()
    print(f"[run_daily] Starting daily pipeline for {run_date}")

    # 1) Try API-Sports schedule
    games_df = fetch_today_games_from_api(run_date)

    # 2) Fallback to dispersion if API returns no games
    if games_df.empty:
        print("[run_daily] Falling back to dispersion-based schedule inference.")
        games_df = fetch_today_games_from_dispersion(run_date)

    if games_df.empty:
        print("[run_daily] No games found even after dispersion fallback; exiting gracefully.")
        write_run_summary(run_date, 0)
        return

    # 3) Build model predictions
    base_preds_path = build_model_predictions(games_df, run_date)

    # 4) Apply market ensemble adjustment (if odds dispersion is present)
    market_preds_path = apply_market_adjustment(base_preds_path, run_date)

    # 5) Run edge picker to produce picks CSV + HTML
    run_picks_pipeline(run_date)

    # 6) Write a simple run summary
    write_run_summary(run_date, len(games_df))

    print("[run_daily] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
