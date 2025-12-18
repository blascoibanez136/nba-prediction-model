from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _infer_y_true_home_win(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Infer actual home win label from backtest joined/scored df.
    Prefers:
      - home_win if present
      - else home_score/away_score
    """
    dropped: Dict[str, int] = {}

    if "home_win" in df.columns:
        y = _to_float(df["home_win"])
        mask = y.isin([0, 1])
        dropped["invalid_home_win"] = int((~mask).sum())
        return y.where(mask, np.nan), dropped

    if "home_score" in df.columns and "away_score" in df.columns:
        hs = _to_float(df["home_score"])
        aw = _to_float(df["away_score"])
        mask = hs.notna() & aw.notna() & (hs != aw)
        dropped["missing_or_tie_score"] = int((~mask).sum())
        y = (hs > aw).astype(float)
        return y.where(mask, np.nan), dropped

    dropped["missing_label"] = int(len(df))
    return pd.Series([np.nan] * len(df)), dropped


def calibration_table(
    df_joined_scored: pd.DataFrame,
    *,
    prob_col: str = "home_win_prob",
    n_buckets: int = 10,
    min_rows_per_bucket: int = 25,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      - bucketed calibration DataFrame
      - summary dict (json-safe)
    """
    n = int(len(df_joined_scored))
    if prob_col not in df_joined_scored.columns:
        return pd.DataFrame(), {
            "n": n,
            "n_used": 0,
            "n_dropped": n,
            "brier": None,
            "logloss": None,
            "dropped_reasons": {"missing_prob_col": n},
        }

    p = _to_float(df_joined_scored[prob_col]).clip(0.0, 1.0)
    y, dropped = _infer_y_true_home_win(df_joined_scored)

    mask = p.notna() & y.notna()
    p_used = p.loc[mask].astype(float).values
    y_used = y.loc[mask].astype(float).values

    n_used = int(mask.sum())
    n_dropped = int(n - n_used)
    dropped["dropped_missing_prob_or_label"] = int(n_dropped)

    if n_used == 0:
        return pd.DataFrame(), {
            "n": n,
            "n_used": 0,
            "n_dropped": n,
            "brier": None,
            "logloss": None,
            "dropped_reasons": dropped,
        }

    # Global scores
    brier = float(np.mean((p_used - y_used) ** 2))
    eps = 1e-15
    p_clip = np.clip(p_used, eps, 1.0 - eps)
    logloss = float(-np.mean(y_used * np.log(p_clip) + (1.0 - y_used) * np.log(1.0 - p_clip)))

    # Buckets
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    bucket_id = np.digitize(p_used, edges, right=True)
    bucket_id = np.clip(bucket_id, 1, n_buckets)

    rows = []
    for b in range(1, n_buckets + 1):
        idx = bucket_id == b
        cnt = int(np.sum(idx))
        if cnt == 0:
            rows.append(
                {
                    "bucket": b,
                    "p_min": float(edges[b - 1]),
                    "p_max": float(edges[b]),
                    "n": 0,
                    "avg_pred_prob": np.nan,
                    "empirical_home_win_rate": np.nan,
                    "gap": np.nan,
                    "abs_gap": np.nan,
                    "brier_bucket": np.nan,
                    "keep": False,
                }
            )
            continue

        pb = p_used[idx]
        yb = y_used[idx]
        avg_p = float(np.mean(pb))
        emp = float(np.mean(yb))
        gap = emp - avg_p

        rows.append(
            {
                "bucket": b,
                "p_min": float(edges[b - 1]),
                "p_max": float(edges[b]),
                "n": cnt,
                "avg_pred_prob": avg_p,
                "empirical_home_win_rate": emp,
                "gap": float(gap),
                "abs_gap": float(abs(gap)),
                "brier_bucket": float(np.mean((pb - yb) ** 2)),
                "keep": bool(cnt >= min_rows_per_bucket),
            }
        )

    cal_df = pd.DataFrame(rows)

    summary = {
        "n": n,
        "n_used": n_used,
        "n_dropped": n_dropped,
        "brier": brier,
        "logloss": logloss,
        "dropped_reasons": dropped,
    }
    return cal_df, summary
