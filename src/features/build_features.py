"""
Feature merging for NBA Pro-Lite model.
"""

from typing import List, Dict, Any


def merge_game_and_odds(
    games: List[Dict[str, Any]],
    odds: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build base game rows, then (if available) attach opponent-adjusted features.
    """
    rows: List[Dict[str, Any]] = []
    for g in games:
        teams = g.get("teams", {})
        home = teams.get("home", {}).get("name")
        away = teams.get("visitors", {}).get("name")
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "game_id": g.get("id") or g.get("gameId"),
            }
        )

    # try to enrich with opponent-adjusted features
    try:
        from .opponent_adjusted import add_opponent_adjusted_features

        rows = add_opponent_adjusted_features(rows)
    except Exception as e:
        # log and keep going — we don't want to break the pipeline
        print(f"Warning: could not add opponent-adjusted features: {e}")

    return rows
