"""
Rolling Backtest & Weekly Report for NBA Pro-Lite
-------------------------------------------------
Computes rolling evaluation metrics (Brier, LogLoss, Spread MAE, Samples)
for the past N days (default 30) using your daily predictions and final
results fetched via the existing calibration helpers.

Outputs:
- outputs/rolling_metrics.csv
- rolling_report.html

Run locally:
    PYTHONPATH=. python src/eval/rolling_eval.py
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Re-use Day 7 helpers
from src.eval.calibration import fetch_results, compute_metrics, load_predictions


# ------------------------------------------------------------
# Utility: generate list of recent dates (today-N → today-1)
# ------------------------------------------------------------
def list_dates(n_days: int = 30):
    today = date.today()
    return [
        (today - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, n_days + 1)
    ][::-1]  # ascending order


# ------------------------------------------------------------
# Try to load predictions (market > base)
# ------------------------------------------------------------
def try_load_predictions(d: str):
    base = f"outputs/predictions_{d}.csv"
    market = f"outputs/predictions_{d}_market.csv"
    for path in (market, base):
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["game_id"] = df["game_id"].astype(str)
            return df
    return pd.DataFrame()


# ------------------------------------------------------------
# Try to load finals via existing fetch_results()
# ------------------------------------------------------------
def try_load_finals(d: str):
    try:
        df = fetch_results(d)
        if not df.empty and "game_id" in df.columns:
            df["game_id"] = df["game_id"].astype(str)
            return df
    except Exception as e:
        print(f"⚠️  Could not load finals for {d}: {e}")
    return pd.DataFrame(columns=["game_id", "home_score", "away_score"])


# ------------------------------------------------------------
# Compute metrics for a single day
# ------------------------------------------------------------
def day_metrics(d: str):
    preds = try_load_predictions(d)
    finals = try_load_finals(d)
    if preds.empty or finals.empty:
        print(f"🛈 Skipping {d}: missing data")
        return None
    metrics, _ = compute_metrics(preds, finals)
    if not metrics:
        return None
    return {
        "date": d,
        "brier": metrics.get("BrierScore"),
        "logloss": metrics.get("LogLoss"),
        "spread_mae": metrics.get("SpreadMAE"),
        "samples": metrics.get("Samples"),
    }


# ------------------------------------------------------------
# Render a simple HTML trend report
# ------------------------------------------------------------
def render_report(df: pd.DataFrame, out_html: str):
    if df.empty:
        html = "<html><body><h1>No metrics available</h1></body></html>"
    else:
        # Sparkline helper
        def spark(series):
            if series.empty:
                return ""
            normalized = (series - series.min()) / (series.max() - series.min() + 1e-9)
            bars = "▁▂▃▄▅▆▇█"
            return "".join(bars[int(v * (len(bars) - 1))] for v in normalized)

        html = f"""
        <html><head><meta charset='utf-8'>
        <style>
        body {{ font-family: system-ui, sans-serif; padding: 16px; }}
        table {{ border-collapse: collapse; margin-top: 12px; }}
        td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
        th {{ background: #f2f2f2; }}
        </style></head><body>
        <h1>NBA Rolling Metrics (Last {len(df)} Days)</h1>
        <table>
        <tr><th>Date</th><th>Brier</th><th>LogLoss</th><th>Spread MAE</th><th>Samples</th></tr>
        {''.join(
            f"<tr><td>{r.date}</td><td>{r.brier:.5f}</td><td>{r.logloss:.5f}</td>"
            f"<td>{r.spread_mae:.3f}</td><td>{int(r.samples)}</td></tr>"
            for r in df.itertuples()
        )}
        </table>
        <h2>Trends (ASCII sparklines)</h2>
        <pre>
Brier:      {spark(df['brier'])}
LogLoss:    {spark(df['logloss'])}
Spread MAE: {spark(df['spread_mae'])}
Samples:    {spark(df['samples'])}
        </pre>
        </body></html>
        """
    with open(out_html, "w") as f:
        f.write(html)
    print(f"✅ Wrote {out_html}")


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main():
    n_days = int(os.getenv("ROLLING_DAYS", 30))
    os.makedirs("outputs", exist_ok=True)

    dates = list_dates(n_days)
    results = []
    for d in dates:
        dm = day_metrics(d)
        if dm:
            results.append(dm)

    df = pd.DataFrame(results)
    out_csv = "outputs/rolling_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv} ({len(df)} rows)")

    render_report(df, "rolling_report.html")


if __name__ == "__main__":
    main()
