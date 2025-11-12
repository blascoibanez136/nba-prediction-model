"""
Evaluation and calibration module for NBA predictions.
Writes:
- outputs/calibration_metrics.json
- outputs/calibration_plot.csv
- calibration_report.html
"""

from __future__ import annotations

import os
import json
from datetime import date
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Helpers: load predictions
# ----------------------------
def load_predictions(date_str: str) -> pd.DataFrame:
    """
    Load predictions for date_str. Prefer market-adjusted file, fallback to base.
    """
    os.makedirs("outputs", exist_ok=True)
    cand = [
        f"outputs/predictions_{date_str}_market.csv",
        f"outputs/predictions_{date_str}.csv",
    ]
    for p in cand:
        if os.path.exists(p):
            print(f"Using predictions: {p}")
            df = pd.read_csv(p)
            # normalize key
            if "game_id" in df.columns:
                df["game_id"] = df["game_id"].astype(str)
            return df
    raise FileNotFoundError(
        f"No predictions found for {date_str}. Looked for: {cand}"
    )


# ----------------------------
# Helpers: fetch results
# ----------------------------
def fetch_results(date_str: str) -> pd.DataFrame:
    """
    Try to obtain final scores for date_str.
    1) Attempt repo ingest (if available)
    2) Optional local fallback CSV: data/_snapshots/finals_<YYYY-MM-DD>.csv
       (columns: game_id, home_score, away_score)
    Returns empty DataFrame if nothing is available (pipeline will stub outputs).
    """
    # (1) Try repo ingest if present
    try:
        from src.ingest.nba_ingest import get_nba_games_by_date  # type: ignore

        resp = get_nba_games_by_date(date_str)
        games = resp.get("response", []) if isinstance(resp, dict) else []
        rows = []
        for g in games:
            # try several common shapes
            gid = g.get("id") or g.get("gameId") or g.get("game_id")
            hs = (
                g.get("scores", {}).get("home")
                if isinstance(g.get("scores"), dict)
                else g.get("home_score")
            )
            as_ = (
                g.get("scores", {}).get("away")
                if isinstance(g.get("scores"), dict)
                else g.get("away_score")
            )
            # accept only completed games with both scores
            if gid is not None and hs is not None and as_ is not None:
                rows.append(
                    {"game_id": str(gid), "home_score": int(hs), "away_score": int(as_)}
                )
        if rows:
            df = pd.DataFrame(rows)
            df["game_id"] = df["game_id"].astype(str)
            print(f"Fetched {len(df)} finals from ingest for {date_str}.")
            return df
        else:
            print("Ingest returned no completed games; trying local fallback...")
    except Exception as e:
        print(f"⚠️  Could not fetch results from ingest: {e}")

    # (2) Fallback to local snapshot, if user provides it
    fallback = f"data/_snapshots/finals_{date_str}.csv"
    if os.path.exists(fallback):
        df = pd.read_csv(fallback)
        # ensure required columns
        for col in ("game_id", "home_score", "away_score"):
            if col not in df.columns:
                print(f"⚠️  Fallback CSV missing column: {col}. Ignoring file.")
                return pd.DataFrame()
        df["game_id"] = df["game_id"].astype(str)
        print(f"Using local fallback finals: {fallback}")
        return df

    # Nothing available
    print("⚠️  No finals available (API blocked/empty and no fallback).")
    return pd.DataFrame(columns=["game_id", "home_score", "away_score"])


# ----------------------------
# Compute metrics (kept as you wrote it)
# ----------------------------
def compute_metrics(preds: pd.DataFrame, results: pd.DataFrame):
    # Guard against missing/empty results
    if results is None or results.empty or ("game_id" not in results.columns):
        print("⚠️  No usable results (empty or missing 'game_id'). Skipping metrics.")
        return {}, pd.DataFrame()

    if preds is None or preds.empty or ("game_id" not in preds.columns):
        print("⚠️  No usable predictions with 'game_id'. Skipping metrics.")
        return {}, pd.DataFrame()

    preds = preds.copy()
    results = results.copy()

    # Coerce merge keys to str
    preds["game_id"] = preds["game_id"].astype(str)
    results["game_id"] = results["game_id"].astype(str)

    df = preds.merge(results, on="game_id", how="inner")
    if df.empty:
        print("⚠️  No rows after merge on game_id. Skipping metrics.")
        return {}, pd.DataFrame()

    # Actual outcome
    df["home_win_actual"] = (df["home_score"] > df["away_score"]).astype(int)

    # pick market or base prob
    pcol = "home_win_prob_market" if "home_win_prob_market" in df.columns else "home_win_prob"
    p = np.clip(df[pcol].astype(float), 1e-6, 1 - 1e-6)
    y = df["home_win_actual"]

    # Metrics
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # Spread MAE
    df["actual_margin"] = df["home_score"] - df["away_score"]
    sp_col = "fair_spread_market" if "fair_spread_market" in df.columns else "fair_spread"
    spread_mae = float(np.mean(np.abs(df["actual_margin"] + df[sp_col].astype(float))))

    # Calibration buckets
    df["bucket"] = pd.cut(p, bins=np.linspace(0, 1, 11), labels=False, include_lowest=True)
    calib_prob_col = "home_win_prob_market" if "home_win_prob_market" in df.columns else "home_win_prob"
    calib = (
        df.groupby("bucket")
          .agg(expected=(calib_prob_col, "mean"),
               actual=("home_win_actual", "mean"),
               n=("home_win_actual", "count"))
          .dropna()
    )

    metrics = {
        "BrierScore": brier,
        "LogLoss": logloss,
        "SpreadMAE": spread_mae,
        "Samples": int(len(df)),
    }
    return metrics, calib


# ----------------------------
# Report writer
# ----------------------------
def write_report(metrics: dict, calib_df: pd.DataFrame, date_str: str) -> None:
    os.makedirs("outputs", exist_ok=True)

    # Save JSON metrics
    with open("outputs/calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save calibration curve data
    calib_out = calib_df.reset_index(drop=True)
    calib_out.to_csv("outputs/calibration_plot.csv", index=False)

    # Simple inline HTML
    html = f"""
<html><head><meta charset="utf-8"><title>Calibration {date_str}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 16px; }}
table {{ border-collapse: collapse; margin-top: 12px; }}
td, th {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #f7f7f7; }}
h1 {{ margin: 0 0 8px 0; }}
</style>
</head>
<body>
  <h1>Calibration — {date_str}</h1>
  <h2>Metrics</h2>
  <table>
    <tr><th>Brier</th><th>LogLoss</th><th>Spread MAE</th><th>Samples</th></tr>
    <tr>
      <td>{metrics.get('BrierScore', float('nan')):.6f}</td>
      <td>{metrics.get('LogLoss', float('nan')):.6f}</td>
      <td>{metrics.get('SpreadMAE', float('nan')):.3f}</td>
      <td>{metrics.get('Samples', 0)}</td>
    </tr>
  </table>

  <h2>Calibration Buckets (Expected vs Actual)</h2>
  <table>
    <tr><th>Bucket</th><th>Expected</th><th>Actual</th><th>n</th></tr>
    {''.join(
        f"<tr><td>{i}</td><td>{row.expected:.3f}</td><td>{row.actual:.3f}</td><td>{int(row.n)}</td></tr>"
        for i, row in calib_out.iterrows()
    )}
  </table>
</body></html>
"""
    with open("calibration_report.html", "w") as f:
        f.write(html)

    print("✅ Wrote: outputs/calibration_metrics.json, outputs/calibration_plot.csv, calibration_report.html")


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    today = os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")
    print(f"Evaluating for {today}")

    # Load predictions (market preferred)
    try:
        preds = load_predictions(today)
    except Exception as e:
        print(f"❌ Could not load predictions: {e}")
        # Graceful stub and exit 0 so workflow doesn't fail
        open("calibration_report.html", "w").write(
            f"<html><body><h1>Calibration</h1><p>No predictions for {today}.</p></body></html>"
        )
        raise SystemExit(0)

    # Get results (may be empty)
    results = fetch_results(today)

    # Compute metrics (handles empty results)
    metrics, calib = compute_metrics(preds, results)
    if not metrics:
        # Graceful stub + exit 0
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/calibration_metrics.json", "w") as f:
            f.write("{}")
        pd.DataFrame(columns=["bucket", "expected", "actual", "n"]).to_csv(
            "outputs/calibration_plot.csv", index=False
        )
        open("calibration_report.html", "w").write(
            f"<html><body><h1>Calibration</h1><p>No final results available for {today} yet.</p></body></html>"
        )
        print("🛈 Wrote stub calibration artifacts (no results yet).")
        raise SystemExit(0)

    write_report(metrics, calib, today)
