from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationSummary:
    brier: Optional[float]
    logloss: Optional[float]
    n: int
    n_used: int
    n_dropped: int
    dropped_reasons: Dict[str, int]


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def infer_actual_home_win(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Robustly infer y_true (home win: 1/0) from common backtest_joined schemas.

    Priority:
      1) explicit boolean/int columns if present
      2) home_score/away_score (or similar)
    """
    dropped: Dict[str, int] = {}

    # Common explicit labels
    for col in ["home_win", "home_won", "actual_home_win", "y_true", "home_win_actual"]:
        if col in df.columns:
            y = df[col]
            y_num = pd.to_numeric(y, errors="coerce")
            ok = y_num.dropna().isin([0, 1]).all()
            if ok:
                return y_num.astype("float"), dropped

    # Score-based inference
    home_score_cols = [c for c in df.columns if c.lower() in ["home_score", "home_points", "pts_home", "home_pts"]]
    away_score_cols = [c for c in df.columns if c.lower() in ["away_score", "away_points", "pts_away", "away_pts"]]
    if home_score_cols and away_score_cols:
        hs = _to_float_series(df[home_score_cols[0]])
        aw = _to_float_series(df[away_score_cols[0]])
        y = (hs > aw).astype("float")
        # drop ties / missing
        mask = hs.notna() & aw.notna() & (hs != aw)
        dropped["missing_or_tie_score"] = int((~mask).sum())
        y = y.where(mask, np.nan)
        return y, dropped

    # Nothing usable
    dropped["missing_label"] = int(len(df))
    return pd.Series([np.nan] * len(df)), dropped


def _bucket_edges(n_buckets: int) -> np.ndarray:
    # e.g. 10 buckets: [0.0,0.1,...,1.0]
    return np.linspace(0.0, 1.0, n_buckets + 1)


def calibration_table(
    df_joined: pd.DataFrame,
    *,
    prob_col: str = "home_win_prob",
    n_buckets: int = 10,
    min_rows_per_bucket: int = 25,
) -> Tuple[pd.DataFrame, CalibrationSummary]:
    """
    Produce reliability buckets + global calibration scores.
    Writes nothing; pure function.
    """
    n = int(len(df_joined))
    dropped_reasons: Dict[str, int] = {}

    if prob_col not in df_joined.columns:
        return (
            pd.DataFrame(),
            CalibrationSummary(
                brier=None,
                logloss=None,
                n=n,
                n_used=0,
                n_dropped=n,
                dropped_reasons={"missing_prob_col": n},
            ),
        )

    p = _to_float_series(df_joined[prob_col]).clip(lower=0.0, upper=1.0)
    y, dropped = infer_actual_home_win(df_joined)
    dropped_reasons.update(dropped)

    mask = p.notna() & y.notna()
    used = df_joined.loc[mask].copy()
    p_used = p.loc[mask].astype("float")
    y_used = y.loc[mask].astype("float")

    n_used = int(mask.sum())
    n_dropped = int(n - n_used)
    dropped_reasons["dropped_missing_prob_or_label"] = int(n_dropped)

    if n_used == 0:
        return (
            pd.DataFrame(),
            CalibrationSummary(
                brier=None,
                logloss=None,
                n=n,
                n_used=0,
                n_dropped=n,
                dropped_reasons=dropped_reasons,
            ),
        )

    # Global scores
    brier = float(np.mean((p_used.values - y_used.values) ** 2))

    eps = 1e-15
    p_clip = np.clip(p_used.values, eps, 1.0 - eps)
    logloss = float(-np.mean(y_used.values * np.log(p_clip) + (1.0 - y_used.values) * np.log(1.0 - p_clip)))

    # Bucketed reliability
    edges = _bucket_edges(n_buckets)
    bucket_id = np.digitize(p_used.values, edges, right=True)
    # digitize returns 0..n_buckets; clamp
    bucket_id = np.clip(bucket_id, 1, n_buckets)

    rows: List[Dict] = []
    for b in range(1, n_buckets + 1):
        idx = bucket_id == b
        cnt = int(np.sum(idx))
        if cnt == 0:
            rows.append(
                dict(
                    bucket=b,
                    p_min=float(edges[b - 1]),
                    p_max=float(edges[b]),
                    n=0,
                    avg_pred_prob=np.nan,
                    empirical_home_win_rate=np.nan,
                    brier_bucket=np.nan,
                    keep=False,
                )
            )
            continue

        pb = p_used.values[idx]
        yb = y_used.values[idx]
        rows.append(
            dict(
                bucket=b,
                p_min=float(edges[b - 1]),
                p_max=float(edges[b]),
                n=cnt,
                avg_pred_prob=float(np.mean(pb)),
                empirical_home_win_rate=float(np.mean(yb)),
                brier_bucket=float(np.mean((pb - yb) ** 2)),
                keep=(cnt >= min_rows_per_bucket),
            )
        )

    cal = pd.DataFrame(rows)
    cal["gap"] = cal["empirical_home_win_rate"] - cal["avg_pred_prob"]
    cal["abs_gap"] = cal["gap"].abs()

    summary = CalibrationSummary(
        brier=brier,
        logloss=logloss,
        n=n,
        n_used=n_used,
        n_dropped=n_dropped,
        dropped_reasons=dropped_reasons,
    )
    return cal, summary
