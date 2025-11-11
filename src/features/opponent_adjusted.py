"""
Opponent-adjusted ratings feature engineering.

This module provides utilities to compute rolling offensive and defensive
ratings for NBA teams using a simple exponential decay to weight recent
games more heavily. Right now it uses synthetic scores so the rest of the
pipeline can run; later we can swap in real box-score data.
"""

from collections import defaultdict
from typing import List, Dict, Any
import numpy as np


def compute_exponential_decay(values: np.ndarray, decay: float = 0.9) -> float:
    """Exponentially weighted average, newest gets biggest weight."""
    if len(values) == 0:
        return float("nan")
    weights = np.power(decay, np.arange(len(values))[::-1])
    return float(np.sum(values * weights) / np.sum(weights))


def add_opponent_adjusted_features(
    rows: List[Dict[str, Any]], lookback: int = 10, decay: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Add rolling off/def ratings to each game row.

    New columns (per game):
      home_off_ra10, home_def_ra10, home_off_adj_ra10, home_def_adj_ra10
      away_off_ra10, away_def_ra10, away_off_adj_ra10, away_def_adj_ra10
    """
    import random

    # per-team scoring history
    history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    enhanced: List[Dict[str, Any]] = []

    def team_metrics(team: str) -> Dict[str, float]:
        logs = history[team][-lookback:]
        if not logs:
            return {
                "off_ra": float("nan"),
                "def_ra": float("nan"),
                "off_adj_ra": float("nan"),
                "def_adj_ra": float("nan"),
            }
        off_scores = np.array([g["team_score"] for g in logs], dtype=float)
        def_scores = np.array([g["opponent_score"] for g in logs], dtype=float)
        off_ra = compute_exponential_decay(off_scores, decay)
        def_ra = compute_exponential_decay(def_scores, decay)
        # placeholder: adjusted = raw
        return {
            "off_ra": off_ra,
            "def_ra": def_ra,
            "off_adj_ra": off_ra,
            "def_adj_ra": def_ra,
        }

    for row in rows:
        home = row.get("home_team")
        away = row.get("away_team")

        # synthetic scores so we can compute something
        home_score = random.randint(90, 130)
        away_score = random.randint(90, 130)

        history[home].append(
            {"team_score": home_score, "opponent_score": away_score}
        )
        history[away].append(
            {"team_score": away_score, "opponent_score": home_score}
        )

        home_m = team_metrics(home)
        away_m = team_metrics(away)

        new_row = row.copy()
        new_row.update(
            {
                "home_off_ra10": home_m["off_ra"],
                "home_def_ra10": home_m["def_ra"],
                "home_off_adj_ra10": home_m["off_adj_ra"],
                "home_def_adj_ra10": home_m["def_adj_ra"],
                "away_off_ra10": away_m["off_ra"],
                "away_def_ra10": away_m["def_ra"],
                "away_off_adj_ra10": away_m["off_adj_ra"],
                "away_def_adj_ra10": away_m["def_adj_ra"],
            }
        )
        enhanced.append(new_row)

    return enhanced
