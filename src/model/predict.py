"""
Prediction utilities for NBA Pro-Lite model.

Uses models trained and saved by src/model/train_model.py.

Main entry:
    predict_games(games_df) -> DataFrame with:
        home_win_prob, away_win_prob, fair_spread, fair_total

Commit-3 hardened behavior:
- Team name mapping for MODEL ENCODING now goes through TeamIndexMapper:
    raw -> canonical franchise (ingest normalizer) -> exact team_index.json key
  This eliminates false "unseen team" fallbacks like:
    "los angeles clippers" vs "LA Clippers"

- Audit-only unseen-team sidecar:
    outputs/audits/unseen_teams_YYYY-MM-DD.json
  Written only when unseen rows occur.

NOTE:
- No change to model math.
- No change to prediction CSV schemas.
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

from src.utils.team_index_mapper import TeamIndexMapper

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

_team_mapper: TeamIndexMapper | None = None


def _load_models() -> None:
    global _team_index, _win_model, _spread_model, _total_model, _team_mapper
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

    # Mapper loads team_index.json internally and builds canonical->key map
    _team_mapper = TeamIndexMapper(TEAM_INDEX_PATH)
    _team_mapper.load()
    _team_index = _team_mapper.team_index

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


def _validate_team_index_keys() -> None:
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
    assert _team_mapper is not None, "TeamIndexMapper not loaded."
    _validate_team_index_keys()

    n_teams = len(_team_index)
    X = np.zeros((len(df), n_teams + 1), dtype=float)

    unseen_rows = 0
    unseen_details: List[dict] = []

    # Map raw team names -> exact team_index keys (or deterministic fallback)
    home_key = df["home_team"].apply(_team_mapper.to_team_index_key)
    away_key = df["away_team"].apply(_team_mapper.to_team_index_key)

    for i, (home, away) in enumerate(zip(home_key, away_key)):
        hi = _team_index.get(home) if home is not None else None
        ai = _team_index.get(away) if away is not None else None

        if hi is None or ai is None:
            # Unseen team => league-average fallback row
            unseen_rows += 1
            X[i, -1] = 1.0

            missing = []
            if hi is None:
                missing.append(str(home))
            if ai is None:
                missing.append(str(away))

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

    # Instrumentation: detect degenerate constant outputs
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
