"""
Training pipeline for NBA Pro-Lite model using balldontlie historical data.

Workflow:
- Read config/config.yaml for backtest.start_season / backtest.end_season
- Use src.ingest.balldontlie_ingest.fetch_games_for_seasons to pull historical games
- Build simple team-strength features:
    * +1 for home team
    * -1 for away team
    * +1 bias term for home court
- Train:
    * LogisticRegression for home win prob
    * LinearRegression for margin (home - away)
    * LinearRegression for total points
- Save:
    * models/team_index.json
    * models/win_model.pkl
    * models/spread_model.pkl
    * models/total_model.pkl

Run from repo root:

    PYTHONPATH=. python src/model/train_model.py

Requires:
- BALLDONTLIE_API_KEY set in environment (for paid plan) or empty for free (v1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.ingest.balldontlie_ingest import fetch_games_for_seasons

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEAM_INDEX_PATH = MODELS_DIR / "team_index.json"
WIN_MODEL_PATH = MODELS_DIR / "win_model.pkl"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.pkl"
TOTAL_MODEL_PATH = MODELS_DIR / "total_model.pkl"


# ---------------------------------------------------------------------
# Config & utilities
# ---------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = ROOT_DIR / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def fetch_historical_games() -> pd.DataFrame:
    """
    Fetch historical NBA games for the backtest window using balldontlie.
    """
    cfg = _load_config()
    bt_cfg = cfg.get("backtest", {})
    start_season = int(bt_cfg.get("start_season", 2019))
    end_season = int(bt_cfg.get("end_season", 2024))

    print(f"[train_model] Fetching games via balldontlie for seasons {start_season}–{end_season}")
    df = fetch_games_for_seasons(start_season, end_season)

    if df.empty:
        raise RuntimeError(
            "No historical games fetched from balldontlie. "
            "Check BALLDONTLIE_API_KEY and backtest seasons in config/config.yaml."
        )

    print(f"[train_model] Collected {len(df)} games total.")
    return df


def build_team_index(df: pd.DataFrame) -> Dict[str, int]:
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    teams = [t for t in teams if isinstance(t, str)]
    teams = sorted(set(teams))
    index = {team: i for i, team in enumerate(teams)}
    print(f"[train_model] Built team index with {len(index)} teams.")
    return index


def make_team_diff_features(df: pd.DataFrame, team_index: Dict[str, int]) -> np.ndarray:
    """
    Feature encoding:
      - +1 in the column of the home team
      - -1 in the column of the away team
      - +1 in the last column for home court bias

    Shape: (n_games, n_teams + 1)
    """
    n_teams = len(team_index)
    X = np.zeros((len(df), n_teams + 1), dtype=float)

    for i, (home, away) in enumerate(zip(df["home_team"], df["away_team"])):
        hi = team_index.get(home)
        ai = team_index.get(away)
        if hi is None or ai is None:
            # unseen team; leave row zeros aside from bias
            X[i, -1] = 1.0
            continue
        X[i, hi] = 1.0
        X[i, ai] = -1.0
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
