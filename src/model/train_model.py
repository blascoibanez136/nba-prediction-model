"""
Basic training pipeline for NBA Pro-Lite model.

This module:
- Pulls historical NBA games from API-Sports (via src.ingest.nba_ingest).
- Builds simple team-strength features.
- Trains:
    * LogisticRegression for home win probability
    * LinearRegression for margin (home - away)
    * LinearRegression for total points
- Saves models + team index under models/

You can run it as a script:

    PYTHONPATH=. python src/model/train_model.py

Note: This will make MANY API calls over the 2019–2024 window
configured in config/config.yaml. Consider starting with a smaller
date range for your first test run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.ingest.nba_ingest import get_nba_games_by_date

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEAM_INDEX_PATH = MODELS_DIR / "team_index.json"
WIN_MODEL_PATH = MODELS_DIR / "win_model.pkl"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.pkl"
TOTAL_MODEL_PATH = MODELS_DIR / "total_model.pkl"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = ROOT_DIR / "config" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _date_range(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _extract_games_for_date(date_str: str) -> List[Dict]:
    """
    Fetch NBA games for a given date via API-Sports and
    return a list of dicts with teams and final scores.
    """
    raw = get_nba_games_by_date(date_str)
    # API-Sports format: raw["response"] is a list of games
    games = raw.get("response", raw.get("games", []))

    rows = []
    for g in games:
        teams = g.get("teams", {})
        home = teams.get("home", {}) or {}
        away = teams.get("visitors", {}) or {}

        home_name = home.get("name")
        away_name = away.get("name")

        scores = g.get("scores", {})
        home_score = scores.get("home")
        away_score = scores.get("visitors")

        # Some API variants nest scores; handle both patterns
        if isinstance(home_score, dict):
            home_score = home_score.get("points")
        if isinstance(away_score, dict):
            away_score = away_score.get("points")

        if (
            home_name is None
            or away_name is None
            or home_score is None
            or away_score is None
        ):
            # skip games that are not finished or malformed
            continue

        rows.append(
            {
                "game_date": date_str,
                "home_team": home_name,
                "away_team": away_name,
                "home_score": int(home_score),
                "away_score": int(away_score),
            }
        )

    return rows


def fetch_historical_games() -> pd.DataFrame:
    """
    Pull historical NBA games over the backtest window defined in config.yaml.
    Returns a DataFrame with columns:
        game_date, home_team, away_team, home_score, away_score
    """
    cfg = _load_config()
    bt_cfg = cfg.get("backtest", {})
    start_season = int(bt_cfg.get("start_season", 2019))
    end_season = int(bt_cfg.get("end_season", 2024))

    # Rough NBA calendar: Oct 1 of start_season to July 1 of end_season+1
    start_date = datetime(start_season, 10, 1)
    end_date = datetime(end_season + 1, 7, 1)

    all_rows: List[Dict] = []

    print(f"[train_model] Fetching games from {start_date.date()} to {end_date.date()}")

    max_days = int(os.getenv("TRAIN_MAX_DAYS", "0"))  # 0 means no cap
    day_counter = 0

    for d in _date_range(start_date, end_date):
        if max_days and day_counter >= max_days:
            print(f"[train_model] TRAIN_MAX_DAYS={max_days} reached, stopping early.")
            break

        ds = d.strftime("%Y-%m-%d")
        try:
            day_rows = _extract_games_for_date(ds)
            if day_rows:
                all_rows.extend(day_rows)
                print(f"[train_model] {ds}: {len(day_rows)} games")
        except Exception as e:
            print(f"[train_model] Warning: failed to fetch {ds}: {e}")

        day_counter += 1

    if not all_rows:
        raise RuntimeError("No historical games fetched. Check API keys and config.")

    df = pd.DataFrame(all_rows)
    print(f"[train_model] Collected {len(df)} games total.")
    return df


def build_team_index(df: pd.DataFrame) -> Dict[str, int]:
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    teams = [t for t in teams if isinstance(t, str)]
    teams = sorted(set(teams))
    index = {team: i for i, team in enumerate(teams)}
    return index


def make_team_diff_features(
    df: pd.DataFrame, team_index: Dict[str, int]
) -> np.ndarray:
    """
    Create a feature matrix where each row encodes:
        +1 for home team, -1 for away team, and a bias term for home court.
    Shape: (n_games, n_teams + 1)
    """
    n_teams = len(team_index)
    X = np.zeros((len(df), n_teams + 1), dtype=float)

    for i, (home, away) in enumerate(zip(df["home_team"], df["away_team"])):
        hi = team_index.get(home)
        ai = team_index.get(away)
        if hi is None or ai is None:
            continue
        X[i, hi] = 1.0
        X[i, ai] = -1.0
        # last column: home-court bias
        X[i, -1] = 1.0

    return X


def build_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    home_score = df["home_score"].values.astype(float)
    away_score = df["away_score"].values.astype(float)

    y_win = (home_score > away_score).astype(int)  # 1 if home wins
    y_margin = home_score - away_score             # home - away
    y_total = home_score + away_score              # total points

    return y_win, y_margin, y_total


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_and_save_models():
    df = fetch_historical_games()

    team_index = build_team_index(df)
    X = make_team_diff_features(df, team_index)
    y_win, y_margin, y_total = build_targets(df)

    print(f"[train_model] Training set shape: {X.shape}")

    # Win-prob model
    win_model = LogisticRegression(max_iter=1000)
    win_model.fit(X, y_win)
    print("[train_model] Trained LogisticRegression for win probability.")

    # Spread model (margin)
    spread_model = LinearRegression()
    spread_model.fit(X, y_margin)
    print("[train_model] Trained LinearRegression for margin (spread).")

    # Total model
    total_model = LinearRegression()
    total_model.fit(X, y_total)
    print("[train_model] Trained LinearRegression for totals.")

    # Save models + team index
    TEAM_INDEX_PATH.write_text(json.dumps(team_index, indent=2))
    dump(win_model, WIN_MODEL_PATH)
    dump(spread_model, SPREAD_MODEL_PATH)
    dump(total_model, TOTAL_MODEL_PATH)

    print(f"[train_model] Saved team index to {TEAM_INDEX_PATH}")
    print(f"[train_model] Saved win model    to {WIN_MODEL_PATH}")
    print(f"[train_model] Saved spread model to {SPREAD_MODEL_PATH}")
    print(f"[train_model] Saved total model  to {TOTAL_MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_models()
