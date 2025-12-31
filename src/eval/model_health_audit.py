from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# helpers
# ----------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _get_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    return None


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def _logloss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _calibration_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns a calibration table + summary ECE.
    Bins are equal-width on [0,1].
    """
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(p, edges, right=True)
    bin_id = np.clip(bin_id, 1, n_bins)

    rows = []
    ece = 0.0
    n = len(p)

    for b in range(1, n_bins + 1):
        m = bin_id == b
        cnt = int(np.sum(m))
        if cnt == 0:
            rows.append({
                "bin": b,
                "p_min": float(edges[b-1]),
                "p_max": float(edges[b]),
                "n": 0,
                "avg_pred": np.nan,
                "empirical": np.nan,
                "gap": np.nan,
                "abs_gap": np.nan,
                "weight": 0.0,
            })
            continue

        avg_pred = float(np.mean(p[m]))
        emp = float(np.mean(y_true[m]))
        gap = emp - avg_pred
        w = cnt / max(n, 1)
        ece += w * abs(gap)

        rows.append({
            "bin": b,
            "p_min": float(edges[b-1]),
            "p_max": float(edges[b]),
            "n": cnt,
            "avg_pred": avg_pred,
            "empirical": emp,
            "gap": float(gap),
            "abs_gap": float(abs(gap)),
            "weight": float(w),
        })

    table = pd.DataFrame(rows)
    summary = {"n": int(n), "ece": float(ece)}
    return table, summary


def _mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    return {
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _tail_rates(abs_err: np.ndarray, thresholds: list[float]) -> Dict[str, float]:
    out = {}
    for t in thresholds:
        out[f"abs_ge_{t:g}"] = float(np.mean(abs_err >= float(t)))
    return out


# ----------------------------
# main audit
# ----------------------------
def run_audit(
    joined_csv: str,
    *,
    out_dir: str = "outputs",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict:
    df = pd.read_csv(joined_csv)
    if df.empty:
        raise RuntimeError("joined_csv is empty")

    # date filter (optional)
    date_col = _get_date_col(df)
    if (start or end):
        if not date_col:
            raise RuntimeError("start/end provided but no date column found")
        d = pd.to_datetime(df[date_col], errors="coerce")
        df = df.assign(_date=d).dropna(subset=["_date"])
        if start:
            df = df[df["_date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["_date"] <= pd.to_datetime(end)]
        df = df.drop(columns=["_date"]).copy()

    # choose probability column
    prob_col = None
    for c in ["home_win_prob", "home_win_prob_market", "home_win_prob_model", "home_win_prob_model_raw"]:
        if c in df.columns:
            prob_col = c
            break
    if prob_col is None:
        raise RuntimeError("No home win probability column found")

    # required score columns (common)
    hcol = None
    acol = None
    for pair in [("home_score", "away_score"), ("home_final_score", "away_final_score"), ("home_pts", "away_pts")]:
        if pair[0] in df.columns and pair[1] in df.columns:
            hcol, acol = pair
            break
    if hcol is None:
        raise RuntimeError("No (home_score, away_score) columns found (or common aliases)")

    # fair spread / fair total columns
    spread_col = None
    for c in ["fair_spread", "fair_spread_model", "spread_pred"]:
        if c in df.columns:
            spread_col = c
            break

    total_col = None
    for c in ["fair_total", "fair_total_model", "total_pred"]:
        if c in df.columns:
            total_col = c
            break

    # coerce
    hs = _to_num(df[hcol])
    aw = _to_num(df[acol])
    p = _to_num(df[prob_col])
    p = p.clip(1e-6, 1.0 - 1e-6)

    # drop invalid rows
    mask = hs.notna() & aw.notna() & p.notna()
    used = df.loc[mask].copy()

    hs_u = _to_num(used[hcol]).to_numpy(dtype=float)
    aw_u = _to_num(used[acol]).to_numpy(dtype=float)
    p_u = _to_num(used[prob_col]).to_numpy(dtype=float)

    y_true = (hs_u > aw_u).astype(float)

    # calibration
    brier = _brier(y_true, p_u)
    logloss = _logloss(y_true, p_u)
    cal_table, cal_summary = _calibration_table(y_true, p_u, n_bins=10)

    # spread metrics
    spread_metrics = {"available": bool(spread_col)}
    if spread_col:
        s = _to_num(used[spread_col]).to_numpy(dtype=float)
        margin = (hs_u - aw_u)
        # note: fair_spread is HOME line (negative means home favored)
        # actual margin is home - away
        # error convention: (predicted_margin - actual_margin)
        # predicted_margin implied by spread = -fair_spread
        pred_margin = -s
        err = pred_margin - margin
        spread_metrics = {
            "available": True,
            "mae": _mae(err),
            "rmse": _rmse(err),
            "bias_mean": float(np.mean(err)),
            "error_quantiles": _quantiles(err),
            "abs_error_quantiles": _quantiles(np.abs(err)),
            "tail_rates": _tail_rates(np.abs(err), [10, 15]),
            "n": int(len(err)),
        }

    # total metrics
    total_metrics = {"available": bool(total_col)}
    if total_col:
        t = _to_num(used[total_col]).to_numpy(dtype=float)
        actual_total = hs_u + aw_u
        err = t - actual_total
        total_metrics = {
            "available": True,
            "mae": _mae(err),
            "rmse": _rmse(err),
            "bias_mean": float(np.mean(err)),
            "error_quantiles": _quantiles(err),
            "abs_error_quantiles": _quantiles(np.abs(err)),
            "tail_rates": _tail_rates(np.abs(err), [15, 20]),
            "n": int(len(err)),
        }

    # degeneracy checks
    def nunique_nonnull(col: str) -> int:
        if col not in used.columns:
            return -1
        return int(pd.to_numeric(used[col], errors="coerce").dropna().nunique())

    degeneracy = {
        "rows_loaded": int(len(df)),
        "rows_used": int(len(used)),
        "prob_col": prob_col,
        "prob_nunique": nunique_nonnull(prob_col),
        "spread_col": spread_col,
        "spread_nunique": nunique_nonnull(spread_col) if spread_col else None,
        "total_col": total_col,
        "total_nunique": nunique_nonnull(total_col) if total_col else None,
        "score_cols": [hcol, acol],
    }

    report = {
        "input": {
            "joined_csv": joined_csv,
            "date_filter": {"start": start, "end": end, "date_col": date_col},
        },
        "coverage": degeneracy,
        "win_prob": {
            "brier": brier,
            "logloss": logloss,
            "ece": float(cal_summary["ece"]),
            "n": int(cal_summary["n"]),
        },
        "spread": spread_metrics,
        "total": total_metrics,
        "artifacts": {
            "calibration_table_csv": str(Path(out_dir) / "model_audit_calibration_table.csv"),
            "report_json": str(Path(out_dir) / "model_health_audit.json"),
        },
    }

    # write outputs
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cal_table.to_csv(Path(out_dir) / "model_audit_calibration_table.csv", index=False)
    Path(out_dir, "model_health_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    ap = argparse.ArgumentParser("model_health_audit")
    ap.add_argument("--joined", required=True, help="Path to joined backtest file (e.g., outputs/backtest_joined_market.csv)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    report = run_audit(args.joined, out_dir=args.out_dir, start=args.start, end=args.end)
    print(json.dumps(report["win_prob"], indent=2))
    if report["spread"].get("available"):
        print("spread:", report["spread"]["mae"], "bias:", report["spread"]["bias_mean"])
    if report["total"].get("available"):
        print("total:", report["total"]["mae"], "bias:", report["total"]["bias_mean"])
    print(f"[audit] wrote {report['artifacts']['report_json']}")
    print(f"[audit] wrote {report['artifacts']['calibration_table_csv']}")


if __name__ == "__main__":
    main()
