"""
Prediction utilities for NBA Pro-Lite model.

Uses models trained and saved by src/model/train_model.py.

Main entry:
    predict_games(games_df) -> DataFrame with:
        home_win_prob, fair_spread, fair_total
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from joblib import load

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"

TEAM_INDEX_PATH = MODELS_DIR / "team_index.json"
WIN_MODEL_PATH = MODELS_DIR / "win_model.pkl"
SPREAD_MODEL_PATH = MODELS_DIR / "spread_model.pkl"
TOTAL_MODEL_PATH = MODELS_DIR / "total_model.pkl"

_team_index: Dict[str, int] | None = None
_win_model = None
_spread_model = None
_total_model = None


def _load_models():
    global _team_index, _win_model, _spread_model, _total_model
    if _team_index is not None:
        return

    _team_index = json.loads(TEAM_INDEX_PATH.read_text())
    _win_model = load(WIN_MODEL_PATH)
    _spread_model = load(SPREAD_MODEL_PATH)
    _total_model = load(TOTAL_MODEL_PATH)


def _make_team_diff_features(df: pd.DataFrame) -> np.ndarray:
    """
    Same encoding as in train_model.make_team_diff_features:
      +1 for home team, -1 for away team, +1 bias for home court.
    """
    assert _team_index is not None, "Models not loaded. Call _load_models() first."

    n_teams = len(_team_index)
    X = np.zeros((len(df), n_teams + 1), dtype=float)

    for i, (home, away) in enumerate(zip(df["home_team"], df["away_team"])):
        hi = _team_index.get(home)
        ai = _team_index.get(away)

        if hi is None or ai is None:
            # Unseen team: skip; leave row zeros aside from bias
            # (model will default to league-average)
            X[i, -1] = 1.0
            continue

        X[i, hi] = 1.0
        X[i, ai] = -1.0
        X[i, -1] = 1.0

    return X


def predict_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with at least:
        home_team, away_team
    Output: original DataFrame +:
        home_win_prob, away_win_prob,
        fair_spread, fair_total

    Note on fair_spread:
        We model margin = home_score - away_score
        Fair spread from home perspective is -margin.
    """
    _load_models()
    X = _make_team_diff_features(games_df)

    win_probs = _win_model.predict_proba(X)[:, 1]  # P(home wins)
    margins = _spread_model.predict(X)             # expected home - away
    totals = _total_model.predict(X)               # expected total points

    out = games_df.copy()
    out["home_win_prob"] = win_probs
    out["away_win_prob"] = 1.0 - win_probs
    out["fair_spread"] = -margins
    out["fair_total"] = totals

    return out
