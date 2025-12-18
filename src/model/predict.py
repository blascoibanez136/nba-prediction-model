"""
Prediction utilities for NBA Pro-Lite model.

Uses models trained and saved by src/model/train_model.py.

Main entry:
    predict_games(games_df) -> DataFrame with:
        home_win_prob, away_win_prob, fair_spread, fair_total

Critical fix (Commit-3 audit-only):
- Emit instrumentation on unseen teams to prevent silent league-average defaults.
- Writes a deterministic sidecar JSON: outputs/audits/unseen_teams_YYYY-MM-DD.json
  when unseen rows occur.

NOTE:
- This patch does NOT change model math.
- This patch does NOT change prediction CSV schemas.
- Canonicalization behavior remains unchanged (to avoid regressions).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from joblib import load

logger = logging.getLogger(__name__)

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

# Common NBA naming aliases (ingest feeds often use these)
_TEAM_ALIASES = {
    "LA Clippers": "LA Clippers",
    "Los Angeles Clippers": "LA Clippers",
    "L.A. Clippers": "LA Clippers",
    "LA Clipppers": "LA Clippers",  # common typo
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "L.A. Lakers": "Los Angeles Lakers",
    "NY Knicks": "New York Knicks",
    "New York Knicks": "New York Knicks",
    "BKN Nets": "Brooklyn Nets",
    "Brooklyn Nets": "Brooklyn Nets",
    "GS Warriors": "Golden State Warriors",
    "Golden State Warriors": "Golden State Warriors",
    "SA Spurs": "San Antonio Spurs",
    "San Antonio Spurs": "San Antonio Spurs",
    "NO Pelicans": "New Orleans Pelicans",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "OKC Thunder": "Oklahoma City Thunder",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "PHX Suns": "Phoenix Suns",
    "Phoenix Suns": "Phoenix Suns",
    "UTA Jazz": "Utah Jazz",
    "Utah Jazz": "Utah Jazz",
    "WAS Wizards": "Washington Wizards",
    "Washington Wizards": "Washington Wizards",
}


def _load_models() -> None:
    global _team_index, _win_model, _spread_model, _total_model
    if _team_index is not None:
        return

    if not TEAM_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing TEAM_INDEX_PATH: {TEAM_INDEX_PATH}")
    if not WIN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing WIN_MODEL_PATH: {WIN_MODEL_PATH}")
    if not SPREAD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing SPREAD_MODEL_PATH: {SPREAD_MODEL_PATH}")
    if not TOTAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing TOTAL_MODEL_PATH: {TOTAL_MODEL_PATH}")

    _team_index = json.loads(TEAM_INDEX_PATH.read_text(encoding="utf-8"))
    _win_model = load(WIN_MODEL_PATH)
    _spread_model = load(SPREAD_MODEL_PATH)
    _total_model = load(TOTAL_MODEL_PATH)

    logger.info(
        "[predict] loaded models: teams=%d win=%s spread=%s total=%s",
        len(_team_index),
        WIN_MODEL_PATH.name,
        SPREAD_MODEL_PATH.name,
        TOTAL_MODEL_PATH.name,
    )


def _normalize_team_name(name: object) -> Optional[str]:
    """
    Canonicalize team names to match team_index.json keys.

    Strategy:
    1) strip whitespace
    2) apply alias map
    3) if exact key exists -> return
    4) else try case-insensitive match against keys
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None

    # apply alias map first (preserves canonical capitalization)
    s2 = _TEAM_ALIASES.get(s, s)

    # exact match
    if _team_index is not None and s2 in _team_index:
        return s2

    # case-insensitive match to existing keys
    if _team_index is not None:
        target = s2.lower()
        for k in _team_index.keys():
            if k.lower() == target:
                return k

    # no match
    return s2


def _validate_team_index_keys() -> None:
    # Simple sanity check to avoid weird mismatches
    assert _team_index is not None
    if len(_team_index) < 25:
        raise RuntimeError(f"[predict] team_index.json seems wrong (only {len(_team_index)} teams).")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_unseen_sidecar(
    *,
    game_date: str,
    unseen_rows: List[dict],
    team_index_keys: List[str],
) -> None:
    """
    Audit-only sidecar for unseen team events.

    Writes: outputs/audits/unseen_teams_YYYY-MM-DD.json

    Does NOT affect predictions, schemas, or model math.
    """
    audits_dir = ROOT_DIR / "outputs" / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    keys_norm = sorted({str(k).strip() for k in team_index_keys if str(k).strip()})
    payload = {
        "game_date": game_date,
        "unseen_rows_count": len(unseen_rows),
        "unseen_rows": unseen_rows[:500],  # cap for safety
        "team_index_keys_count": len(keys_norm),
        "team_index_keys_sample": keys_norm[:50],
        "team_index_keys_sha256": _sha256_text("|".join(keys_norm)) if keys_norm else None,
    }

    out_path = audits_dir / f"unseen_teams_{game_date}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("[predict] wrote unseen-teams sidecar: %s", out_path.as_posix())


def _make_team_diff_features(df: pd.DataFrame) -> Tuple[np.ndarray, int, List[dict]]:
    """
    Same encoding as in train_model.make_team_diff_features:
      +1 for home team, -1 for away team, +1 bias for home court.

    Returns:
      X, n_unseen_rows, unseen_details
    """
    assert _team_index is not None, "Models not loaded. Call _load_models() first."
    _validate_team_index_keys()

    n_teams = len(_team_index)
    X = np.zeros((len(df), n_teams + 1), dtype=float)

    unseen_rows = 0
    unseen_details: List[dict] = []

    # normalize to canonical team keys
    home_norm = df["home_team"].apply(_normalize_team_name)
    away_norm = df["away_team"].apply(_normalize_team_name)

    for i, (home, away) in enumerate(zip(home_norm, away_norm)):
        hi = _team_index.get(home) if home is not None else None
        ai = _team_index.get(away) if away is not None else None

        if hi is None or ai is None:
            # Unseen team => leave row zeros aside from bias (league-average fallback)
            unseen_rows += 1
            X[i, -1] = 1.0

            missing = []
            if hi is None:
                missing.append(str(home))
            if ai is None:
                missing.append(str(away))

            # Capture raw + normalized + which team keys were missing
            try:
                home_raw = str(df.iloc[i]["home_team"])
                away_raw = str(df.iloc[i]["away_team"])
            except Exception:
                home_raw = ""
                away_raw = ""

            unseen_details.append(
                {
                    "row_index": int(i),
                    "home_team_raw": home_raw,
                    "away_team_raw": away_raw,
                    "home_team_norm": None if home is None else str(home),
                    "away_team_norm": None if away is None else str(away),
                    "missing_team_keys": missing,
                }
            )
            continue

        X[i, hi] = 1.0
        X[i, ai] = -1.0
        X[i, -1] = 1.0

    return X, unseen_rows, unseen_details


def _infer_game_date(games_df: pd.DataFrame) -> str:
    """
    Best-effort deterministic date string for audit filenames.
    Prefers 'game_date' then 'date', else 'unknown_date'.
    """
    for col in ("game_date", "date"):
        if col in games_df.columns and len(games_df) > 0:
            v = str(games_df[col].iloc[0]).strip()
            if v:
                return v
    return "unknown_date"


def predict_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with at least:
        home_team, away_team

    Output: original DataFrame +:
        home_win_prob, away_win_prob, fair_spread, fair_total

    Note on fair_spread:
        We model margin = home_score - away_score
        Fair spread from home perspective is -margin.
    """
    if games_df is None or games_df.empty:
        raise RuntimeError("[predict] games_df is empty")

    required = {"home_team", "away_team"}
    missing = required - set(games_df.columns)
    if missing:
        raise RuntimeError(f"[predict] Missing required columns: {sorted(missing)}")

    _load_models()

    X, unseen_rows, unseen_details = _make_team_diff_features(games_df)

    n = len(games_df)
    if unseen_rows > 0:
        frac = unseen_rows / max(n, 1)
        logger.warning(
            "[predict] unseen teams in %d/%d rows (%.1f%%). Those rows will default to league-average; "
            "check canonicalization upstream.",
            unseen_rows,
            n,
            100.0 * frac,
        )

        # Audit-only: write unseen-team details sidecar so this is debuggable
        try:
            game_date = _infer_game_date(games_df)
            assert _team_index is not None
            _write_unseen_sidecar(
                game_date=game_date,
                unseen_rows=unseen_details,
                team_index_keys=list(_team_index.keys()),
            )
        except Exception as e:
            # Never break predictions for audit failures
            logger.error("[predict] failed to write unseen sidecar: %r", e)

    # Predict
    win_probs = _win_model.predict_proba(X)[:, 1]  # P(home wins)
    margins = _spread_model.predict(X)             # expected home - away
    totals = _total_model.predict(X)               # expected total points

    out = games_df.copy()
    out["home_win_prob"] = win_probs
    out["away_win_prob"] = 1.0 - win_probs
    out["fair_spread"] = -margins
    out["fair_total"] = totals

    # Instrumentation: detect degenerate constant outputs (this is what bit us)
    nun_p = int(pd.Series(out["home_win_prob"]).nunique(dropna=True))
    nun_s = int(pd.Series(out["fair_spread"]).nunique(dropna=True))
    nun_t = int(pd.Series(out["fair_total"]).nunique(dropna=True))

    if nun_p <= 1:
        logger.error("[predict] home_win_prob appears constant (nunique=%d). This will break edge logic.", nun_p)
    if nun_s <= 1:
        logger.error("[predict] fair_spread appears constant (nunique=%d). ATS will be unsafe.", nun_s)
    if nun_t <= 1:
        logger.warning("[predict] fair_total appears constant (nunique=%d). Totals will be weak/unsafe.", nun_t)

    return out
