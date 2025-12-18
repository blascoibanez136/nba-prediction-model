from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PicksPolicy:
    # Core thresholds
    prob_floor: float = 0.62

    # Calibration gating (optional if calibration file exists)
    require_calibration_keep: bool = True
    max_abs_gap: float = 0.08

    # Safety caps
    max_picks_per_day: int = 3
    min_games_for_picks: int = 2

    # Bucket settings (must match backtest_calibration.csv generation)
    n_buckets: int = 10


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _bucket_edges(n_buckets: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_buckets + 1)


def _assign_bucket(p: float, edges: np.ndarray) -> int:
    # p in [0,1]; returns 1..n_buckets
    b = int(np.digitize([p], edges, right=True)[0])
    return int(np.clip(b, 1, len(edges) - 1))


def load_calibration_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"bucket", "p_min", "p_max", "n", "avg_pred_prob", "empirical_home_win_rate", "gap", "abs_gap", "keep"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[picks] calibration table missing columns: {sorted(missing)}")
    df = df.copy()
    df["bucket"] = pd.to_numeric(df["bucket"], errors="coerce").astype("Int64")
    df["keep"] = df["keep"].astype(bool)
    df["abs_gap"] = _to_float(df["abs_gap"])
    df["gap"] = _to_float(df["gap"])
    return df


def generate_conservative_picks(
    preds_df: pd.DataFrame,
    *,
    calibration_df: Optional[pd.DataFrame] = None,
    policy: PicksPolicy = PicksPolicy(),
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      picks_df: subset of preds_df with pick fields (may be empty)
      audit: json-safe dict explaining counts and reasons
    """
    if preds_df is None or preds_df.empty:
        raise RuntimeError("[picks] preds_df is empty")

    required = {"home_team", "away_team"}
    missing = required - set(preds_df.columns)
    if missing:
        raise RuntimeError(f"[picks] preds_df missing required columns: {sorted(missing)}")

    if "home_win_prob" not in preds_df.columns:
        raise RuntimeError("[picks] preds_df missing home_win_prob (required for picks)")

    df = preds_df.copy()

    # Ensure game_date exists (historical runner uses game_date; some outputs may use date)
    if "game_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "game_date"})
    if "game_date" not in df.columns:
        # still produce picks, but date will be blank
        df["game_date"] = ""

    # Safety: tiny slates should default to no picks
    n_games = int(len(df))
    if n_games < policy.min_games_for_picks:
        audit = {
            "n_games": n_games,
            "n_candidates": 0,
            "n_picks": 0,
            "policy": policy.__dict__,
            "reason": f"min_games_for_picks={policy.min_games_for_picks} not met",
        }
        return df.head(0), audit

    # Normalize probability
    df["home_win_prob"] = _to_float(df["home_win_prob"]).clip(0.0, 1.0)

    # Attach calibration info if available
    edges = _bucket_edges(policy.n_buckets)
    cal_map: Dict[int, Dict] = {}

    if calibration_df is not None and not calibration_df.empty:
        for _, r in calibration_df.iterrows():
            b = int(r["bucket"])
            cal_map[b] = {
                "keep": bool(r["keep"]),
                "gap": float(r["gap"]) if pd.notna(r["gap"]) else math.nan,
                "abs_gap": float(r["abs_gap"]) if pd.notna(r["abs_gap"]) else math.nan,
                "n": int(r["n"]) if pd.notna(r["n"]) else 0,
                "p_min": float(r["p_min"]) if pd.notna(r["p_min"]) else math.nan,
                "p_max": float(r["p_max"]) if pd.notna(r["p_max"]) else math.nan,
            }

    # Build candidates + reasons
    reasons: List[str] = []
    bucket_list: List[Optional[int]] = []
    keep_list: List[Optional[bool]] = []
    abs_gap_list: List[Optional[float]] = []
    gap_list: List[Optional[float]] = []

    for p in df["home_win_prob"].tolist():
        if pd.isna(p):
            bucket_list.append(None)
            keep_list.append(None)
            abs_gap_list.append(None)
            gap_list.append(None)
            continue
        b = _assign_bucket(float(p), edges)
        bucket_list.append(b)
        if b in cal_map:
            keep_list.append(cal_map[b]["keep"])
            abs_gap_list.append(cal_map[b]["abs_gap"])
            gap_list.append(cal_map[b]["gap"])
        else:
            keep_list.append(None)
            abs_gap_list.append(None)
            gap_list.append(None)

    df["cal_bucket"] = bucket_list
    df["cal_keep"] = keep_list
    df["cal_gap"] = gap_list
    df["cal_abs_gap"] = abs_gap_list

    # Gate 1: prob floor
    df["gate_prob_floor"] = df["home_win_prob"] >= policy.prob_floor

    # Gate 2: calibration keep (if enabled and calibration present)
    if policy.require_calibration_keep and calibration_df is not None and not calibration_df.empty:
        df["gate_cal_keep"] = df["cal_keep"].fillna(False)
    else:
        # if no calibration data, do not fail picks; just mark as "unknown"
        df["gate_cal_keep"] = True

    # Gate 3: abs gap (if calibration present)
    if calibration_df is not None and not calibration_df.empty:
        df["gate_cal_gap"] = df["cal_abs_gap"].fillna(0.0) <= policy.max_abs_gap
    else:
        df["gate_cal_gap"] = True

    # Combine
    df["is_pick"] = df["gate_prob_floor"] & df["gate_cal_keep"] & df["gate_cal_gap"]

    candidates = df[df["is_pick"]].copy()

    # Rank candidates: highest prob first
    candidates = candidates.sort_values(["home_win_prob", "home_team", "away_team"], ascending=[False, True, True])

    # Cap picks/day
    picks = candidates.head(policy.max_picks_per_day).copy()

    # Add pick fields
    picks["pick_type"] = "ML"
    picks["pick_side"] = "HOME"
    picks["confidence"] = picks["home_win_prob"]
    picks["policy_prob_floor"] = policy.prob_floor
    picks["policy_max_abs_gap"] = policy.max_abs_gap
    picks["policy_require_cal_keep"] = policy.require_calibration_keep
    picks["policy_max_picks_per_day"] = policy.max_picks_per_day

    # Reasons for transparency
    def _reason_row(r) -> str:
        parts = []
        parts.append(f"p>={policy.prob_floor:.2f}")
        if calibration_df is not None and not calibration_df.empty:
            parts.append("cal_keep" if bool(r.get("cal_keep")) else "cal_keep=FALSE")
            parts.append(f"abs_gap<={policy.max_abs_gap:.2f}")
        else:
            parts.append("calibration=missing")
        return ";".join(parts)

    picks["reason"] = picks.apply(_reason_row, axis=1)

    # Output selection columns (keep it stable + informative)
    base_cols = [
        "game_date",
        "home_team",
        "away_team",
        "home_win_prob",
        "fair_spread" if "fair_spread" in picks.columns else None,
        "fair_total" if "fair_total" in picks.columns else None,
        "cal_bucket",
        "cal_keep",
        "cal_abs_gap",
        "pick_type",
        "pick_side",
        "confidence",
        "reason",
    ]
    base_cols = [c for c in base_cols if c is not None and c in picks.columns]

    picks_out = picks[base_cols].copy()

    # Audit
    audit = {
        "n_games": n_games,
        "n_candidates": int(len(candidates)),
        "n_picks": int(len(picks_out)),
        "policy": policy.__dict__,
        "calibration_present": bool(calibration_df is not None and not calibration_df.empty),
        "notes": [
            "Audit-only conservative picks layer; does not alter predictions.",
            "Pick selection is HOME ML only for Pro-Lite (safe default).",
        ],
    }

    return picks_out, audit
