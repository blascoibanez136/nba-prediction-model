"""
run_daily.py

End-to-end daily NBA Pro-Lite pipeline.

Schedule logic (in order of preference):
1) Try to fetch today's NBA games from balldontlie.io.
2) If none are found, fall back to a pre-loaded official schedule CSV:
       data/schedules/nba_schedule.csv
3) If still no games, fall back to inferring the slate from odds_dispersion_latest.csv.

Then:
- Use the trained models to generate predictions (predict_games).
- Add merge_key for joining with odds.
- Apply market_ensemble (if odds dispersion file available).
- Run edge_picker to produce:
      - outputs/picks_<YYYY-MM-DD>.csv
      - picks_report.html
- Write run_summary.md.

Environment:
- RUN_DATE            : optional override (YYYY-MM-DD); default is today's date (UTC).
- BALLDONTLIE_API_KEY : optional API key for balldontlie.io (Authorization: Bearer ...).
"""

from __future__ import annotations

import os
from datetime import date
from typing import List, Dict, Any

import pandas as pd
import requests

from src.model.predict import predict_games
from src.model.market_ensemble import apply_market_ensemble
from src.eval.edge_picker import main as run_edge_picker, _merge_key as _ep_merge_key


BALD_BASE_URL = "https://api.balldontlie.io/v1"
BALD_API_KEY = os.getenv("BALLDONTLIE_API_KEY")


def _get_run_date() -> str:
    return os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")


# -----------------------
# Schedule fetchers
# -----------------------
def fetch_today_games_from_balldontlie(run_date: str) -> pd.DataFrame:
    """
    Fetch today's NBA games from balldontlie.io /games endpoint.

    Endpoint docs (v1):
        GET /games?dates[]=YYYY-MM-DD&per_page=100

    Returns DataFrame with:
        game_id, game_date, home_team, away_team
    (May be empty if the API has no games for that date.)
    """
    params = {
        "dates[]": run_date,
        "per_page": 100,
    }
    headers: Dict[str, str] = {}
    if BALD_API_KEY:
        headers["Authorization"] = f"Bearer {BALD_API_KEY}"

    url = f"{BALD_BASE_URL}/games"
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[run_daily] balldontlie schedule fetch failed for {run_date}: {e}")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    data = resp.json().get("data", [])
    rows: List[Dict[str, Any]] = []

    for g in data:
        gid = g.get("id")
        home = g.get("home_team", {}) or {}
        away = g.get("visitor_team", {}) or {}

        home_name = home.get("full_name") or home.get("name")
        away_name = away.get("full_name") or away.get("name")

        if not home_name or not away_name:
            continue

        d = g.get("date") or ""
        # Typically ISO like "2025-12-07T00:00:00.000Z"
        game_date = d[:10] if d else run_date

        rows.append(
            {
                "game_id": gid,
                "game_date": game_date,
                "home_team": home_name,
                "away_team": away_name,
            }
        )

    df = pd.DataFrame(rows, columns=["game_id", "game_date", "home_team", "away_team"])
    if df.empty:
        print(f"[run_daily] balldontlie returned no games for {run_date}.")
    else:
        print(f"[run_daily] Fetched {len(df)} games from balldontlie for {run_date}.")
    return df


def fetch_today_games_from_schedule_csv(
    run_date: str,
    csv_path: str = "data/schedules/nba_schedule.csv",
) -> pd.DataFrame:
    """
    Fallback: use a pre-loaded official schedule CSV.

    CSV is expected to have at least:
        game_date, home_team, away_team
    Optional:
        game_id

    Returns DataFrame with:
        game_id, game_date, home_team, away_team
    """
    if not os.path.exists(csv_path):
        print(f"[run_daily] No schedule CSV found at {csv_path}.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    sched = pd.read_csv(csv_path)
    required = {"game_date", "home_team", "away_team"}
    missing = required - set(sched.columns)
    if missing:
        print(f"[run_daily] schedule CSV missing columns: {missing}")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    day = sched[sched["game_date"] == run_date].copy()
    if day.empty:
        print(f"[run_daily] schedule CSV has no rows for {run_date}.")
        return pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team"])

    if "game_id" not in day.columns:
        # Synthetic ID if not provided
        day["game_id"] = day.apply(
            lambda r: f"{r['home_team']} vs {r['away_team']} {r['game_date']}", axis=1
        )

    out = day[["game_id", "game_date", "home_team", "away_team"]].copy()
    print(f"[run_daily] Loaded {len(out)} games from schedule CSV for {run_date}.")
    return out


def fetch_today_games_from_dispersion(run_date: str) -> pd.DataFrame:
    """
    Last-resort fallback: infer today's slate from outputs/odds_dispersion_latest.csv.

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
    if "merge_key" in day.columns:
        day = day.sort_values("merge_key").drop_duplicates("merge_key")
        games_df = day[["home_team", "away_team", "game_date", "merge_key"]].copy()
        games_df.rename(columns={"merge_key": "game_id"}, inplace=True)
    else:
        day = day.sort_values(["home_team", "away_team"]).drop_duplicates(
            ["home_team", "away_team"]
        )
        games_df = day[["home_team", "away_team", "game_date"]].copy()
        games_df["game_id"] = games_df.apply(
            lambda r: f"{r['home_team']} vs {r['away_team']} {r['game_date']}", axis=1
        )

    print(f"[run_daily] Inferred {len(games_df)} games from dispersion for {run_date}.")
    return games_df[["game_id", "game_date", "home_team", "away_team"]]


# -----------------------
# Predictions + ensemble
# -----------------------
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
    # Ensure downstream code sees the same RUN_DATE
    os.environ["RUN_DATE"] = run_date
    run_edge_picker()
    print("[run_daily] Edge picker completed.")


def write_run_summary(run_date: str, n_games: int) -> None:
    text = f"# NBA daily run\n\nDate: {run_date}\nGames predicted: {n_games}\n"
    with open("run_summary.md", "w") as f:
        f.write(text)
    print("[run_daily] Wrote run_summary.md")


# -----------------------
# main
# -----------------------
def main():
    run_date = _get_run_date()
    print(f"[run_daily] Starting daily pipeline for {run_date}")

    # Keep RUN_DATE consistent for any module that reads it
    os.environ["RUN_DATE"] = run_date

    # 1) Try balldontlie schedule
    games_df = fetch_today_games_from_balldontlie(run_date)

    # 2) Fallback to CSV schedule
    if games_df.empty:
        print("[run_daily] Falling back to schedule CSV.")
        games_df = fetch_today_games_from_schedule_csv(run_date)

    # 3) Fallback to dispersion-based slate inference
    if games_df.empty:
        print("[run_daily] Falling back to dispersion-based schedule inference.")
        games_df = fetch_today_games_from_dispersion(run_date)

    if games_df.empty:
        print("[run_daily] No games found after all schedule methods; exiting gracefully.")
        write_run_summary(run_date, 0)
        return

    # 4) Build model predictions
    base_preds_path = build_model_predictions(games_df, run_date)

    # 5) Apply market ensemble adjustment (if odds dispersion is present)
    _ = apply_market_adjustment(base_preds_path, run_date)

    # 6) Run edge picker to produce picks CSV + HTML
    run_picks_pipeline(run_date)

    # 7) Write a simple run summary
    write_run_summary(run_date, len(games_df))

    print("[run_daily] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
