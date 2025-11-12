# --- Compute metrics ---
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
    # use the same prob column present in df
    calib_prob_col = "home_win_prob_market" if "home_win_prob_market" in df.columns else "home_win_prob"
    calib = (
        df.groupby("bucket")
          .agg(expected=(calib_prob_col, "mean"), actual=("home_win_actual", "mean"), n=("home_win_actual", "count"))
          .dropna()
    )

    metrics = {
        "BrierScore": brier,
        "LogLoss": logloss,
        "SpreadMAE": spread_mae,
        "Samples": int(len(df)),
    }
    return metrics, calib


# --- CLI entrypoint ---
if __name__ == "__main__":
    today = os.getenv("RUN_DATE") or date.today().strftime("%Y-%m-%d")
    print(f"Evaluating for {today}")
    try:
        preds = load_predictions(today)
    except Exception as e:
        print(f"❌ Could not load predictions: {e}")
        # Write a stub and exit 0 so workflow doesn't fail
        open("calibration_report.html", "w").write(
            f"<html><body><h1>Calibration</h1><p>No predictions for {today}.</p></body></html>"
        )
        raise SystemExit(0)

    results = fetch_results(today)
    metrics, calib = compute_metrics(preds, results)
    if not metrics:
        # Graceful stub + exit 0
        os.makedirs("outputs", exist_ok=True)
        open("outputs/calibration_metrics.json", "w").write("{}")
        pd.DataFrame(columns=["bucket", "expected", "actual", "n"]).to_csv("outputs/calibration_plot.csv", index=False)
        open("calibration_report.html", "w").write(
            f"<html><body><h1>Calibration</h1><p>No final results available for {today} yet.</p></body></html>"
        )
        print("🛈 Wrote stub calibration artifacts (no results yet).")
        raise SystemExit(0)

    write_report(metrics, calib, today)

