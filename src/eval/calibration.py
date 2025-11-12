"""
Evaluation and calibration module for NBA predictions.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import date
from typing import Tuple

# --- Load predictions ---
def load_predictions(date_str: str) -> pd.DataFrame:
    base = f"outputs/predictions_{date_str}.csv"
    market = f"outputs/predictions_{date_str}_market.csv"
    if os.path.exists(market):
        print(f"Using market-adjusted predictions: {market}")
        return pd.read_csv(market)
    elif os.path.exists(base):
        print(f"Using base predictions: {base}")
        return pd.read_csv(base)
    else:
        raise FileNotFoundError("No predictions file found.")

# --- Fetch results ---
def fetch_results(date_str: str) -> pd.DataFrame:
    """
    Fetch NBA final scores using your existing ingest layer.
    Falls back to local CSV if API not available.
    """
    try:
        from src.ingest.nba_ingest import get_nba_games_by_date
        data = get_nba_games_by_date(date_str)
        games = data.get("response", [])
        rows = []
        for g in games:
            if g.get("scores"):
                home = g["teams"]["home"]["name"]
                away = g["teams"]["away"]["name"]
                rows.append({
                    "game_id": str(g["id"]),
                    "home_team": home,
                    "away_team": away,
                    "home_score": g["scores"]["home"]["points"],
                    "away_score": g["scores"]["away"]["points"]
                })
        return pd.DataFrame(rows)
    except Exception as e:
        print("⚠️ Could not fetch results:", e)
        local_path = f"data/results_{date_str}.csv"
        if os.path.exists(local_path):
            print(f"Using fallback {local_path}")
            return pd.read_csv(local_path)
        else:
            return pd.DataFrame()

# --- Compute metrics ---
def compute_metrics(preds: pd.DataFrame, results: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    preds["game_id"] = preds["game_id"].astype(str)
    results["game_id"] = results["game_id"].astype(str)

    df = preds.merge(results, on="game_id", how="inner")
    if df.empty:
        return {}, pd.DataFrame()

    df["home_win_actual"] = (df["home_score"] > df["away_score"]).astype(int)
    p = np.clip(df.get("home_win_prob_market", df["home_win_prob"]), 1e-6, 1 - 1e-6)
    y = df["home_win_actual"]

    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    df["actual_margin"] = df["home_score"] - df["away_score"]
    if "fair_spread_market" in df.columns:
        spread_mae = float(np.mean(np.abs(df["actual_margin"] + df["fair_spread_market"])))
    else:
        spread_mae = float(np.mean(np.abs(df["actual_margin"] + df["fair_spread"])))

    # Calibration buckets
    df["bucket"] = pd.cut(p, bins=np.linspace(0, 1, 11), labels=False, include_lowest=True)
    calib = df.groupby("bucket").agg(expected=("home_win_prob_market", "mean"), actual=("home_win_actual", "mean")).dropna()

    metrics = {
        "BrierScore": brier,
        "LogLoss": logloss,
        "SpreadMAE": spread_mae,
        "Samples": int(len(df)),
    }
    return metrics, calib

# --- Write report ---
def write_report(metrics: dict, calib_df: pd.DataFrame, date_str: str):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    calib_df.to_csv("outputs/calibration_plot.csv", index=True)

    html = f"""
    <html><body>
    <h1>NBA Calibration Report — {date_str}</h1>
    <pre>{json.dumps(metrics, indent=2)}</pre>
    <h3>Calibration Curve</h3>
    {calib_df.to_html()}
    </body></html>
    """
    with open("calibration_report.html", "w") as f:
        f.write(html)
    print("✅ Wrote calibration_report.html")

# --- CLI entrypoint ---
if __name__ == "__main__":
    today = os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")
    print(f"Evaluating for {today}")
    preds = load_predictions(today)
    results = fetch_results(today)
    metrics, calib = compute_metrics(preds, results)
    if not metrics:
        print("No results yet; writing stub report.")
        open("calibration_report.html", "w").write("<html><body><h1>No results yet</h1></body></html>")
    else:
        write_report(metrics, calib, today)
