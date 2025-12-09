"""
Backtesting engine for NBA Pro-Lite.

Usage (from repo root):

    python -m src.eval.backtest --start 2025-10-01 --end 2025-12-31 \
        --results data/results/results_master.csv

What it does:

- Loads final scores from results_master.csv
- Loads daily predictions between start and end dates:
    * Prefer: outputs/predictions_YYYY-MM-DD_market.csv
    * Fallback: outputs/predictions_YYYY-MM-DD.csv
- Joins predictions with results on merge_key
- Computes:
    * Brier score for:
        - home_win_prob (model-only)
        - home_win_prob_market (market ensemble)
    * Spread MAE for:
        - fair_spread
        - fair_spread_market
    * Simple calibration buckets

Outputs:

- outputs/backtest_summary_START_END.md
- outputs/backtest_calibration_START_END.csv

You can extend this later to compute ROI from picks once results+odds
are fully standardized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _date_range(start: str, end: str) -> List[str]:
    d0 = _parse_date(start)
    d1 = _parse_date(end)
    days: List[str] = []
    cur = d0
    while cur <= d1:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    required = {"merge_key", "game_date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"results_master.csv is missing required columns: {sorted(missing)}\n"
            f"Expected at least: {sorted(required)}"
        )

    # Ensure correct types
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return df


def load_predictions_for_range(start: str, end: str) -> pd.DataFrame:
    """
    Load predictions_{date}_market.csv (preferred) or predictions_{date}.csv
    for each date in the range.
    """
    frames: List[pd.DataFrame] = []

    for date_str in _date_range(start, end):
        market_path = OUTPUTS_DIR / f"predictions_{date_str}_market.csv"
        base_path = OUTPUTS_DIR / f"predictions_{date_str}.csv"

        if market_path.exists():
            df = pd.read_csv(market_path)
            df["game_date"] = date_str
            frames.append(df)
        elif base_path.exists():
            df = pd.read_csv(base_path)
            df["game_date"] = date_str
            frames.append(df)
        else:
            # No predictions for this date; skip silently.
            continue

    if not frames:
        raise FileNotFoundError(
            f"No prediction files found between {start} and {end}.\n"
            f"Looked for predictions_YYYY-MM-DD[_market].csv under outputs/."
        )

    preds = pd.concat(frames, ignore_index=True)

    if "merge_key" not in preds.columns:
        raise ValueError("Predictions files are missing 'merge_key' column.")

    # Normalize date formats
    preds["game_date"] = pd.to_datetime(preds["game_date"]).dt.strftime("%Y-%m-%d")

    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BrierResult:
    name: str
    brier: float
    n: int


def brier_score(probs: pd.Series, outcomes: pd.Series, name: str) -> Optional[BrierResult]:
    mask = probs.notna() & outcomes.notna()
    if mask.sum() == 0:
        return None

    p = probs[mask].astype(float).clip(0.0, 1.0)
    y = outcomes[mask].astype(float)
    b = float(((p - y) ** 2).mean())
    return BrierResult(name=name, brier=b, n=int(mask.sum()))


@dataclass
class SpreadMAE:
    name: str
    mae: float
    n: int


def spread_mae(fair_spread: pd.Series, margin: pd.Series, name: str) -> Optional[SpreadMAE]:
    mask = fair_spread.notna() & margin.notna()
    if mask.sum() == 0:
        return None

    fs = fair_spread[mask].astype(float)
    m = margin[mask].astype(float)
    mae = float((fs - m).abs().mean())
    return SpreadMAE(name=name, mae=mae, n=int(mask.sum()))


def calibration_table(
    probs: pd.Series,
    outcomes: pd.Series,
    bucket_width: float = 0.05,
) -> pd.DataFrame:
    """
    Build a simple calibration table: bucket, expected, actual, n.

    Example:
        bucket  expected  actual    n
        0.45    0.45      0.48      50
        0.50    0.50      0.51      60
    """
    mask = probs.notna() & outcomes.notna()
    if mask.sum() == 0:
        return pd.DataFrame(columns=["bucket", "expected", "actual", "n"])

    p = probs[mask].astype(float).clip(0.0, 1.0)
    y = outcomes[mask].astype(float)

    # Assign buckets by rounding to nearest bucket_width
    buckets = (p / bucket_width).round() * bucket_width
    df = pd.DataFrame({"bucket": buckets, "p": p, "y": y})

    grouped = df.groupby("bucket")
    out = grouped.agg(expected=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()
    out = out.sort_values("bucket").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Backtest driver
# ---------------------------------------------------------------------------

def run_backtest(start: str, end: str, results_path: str) -> None:
    results_file = REPO_ROOT / results_path
    results = load_results(results_file)
    preds = load_predictions_for_range(start, end)

    # Join predictions with results on merge_key
    merged = preds.merge(
        results[["merge_key", "home_score", "away_score"]],
        on="merge_key",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No overlapping games between predictions and results.\n"
            "Check that merge_key formatting matches in both files."
        )

    merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype(float)
    merged["margin"] = merged["home_score"] - merged["away_score"]

    # Compute metrics
    brier_results: List[BrierResult] = []
    spread_results: List[SpreadMAE] = []

    if "home_win_prob" in merged.columns:
        br = brier_score(merged["home_win_prob"], merged["home_win"], "home_win_prob")
        if br:
            brier_results.append(br)

    if "home_win_prob_market" in merged.columns:
        brm = brier_score(merged["home_win_prob_market"], merged["home_win"], "home_win_prob_market")
        if brm:
            brier_results.append(brm)

    if "fair_spread" in merged.columns:
        sm = spread_mae(merged["fair_spread"], merged["margin"], "fair_spread")
        if sm:
            spread_results.append(sm)

    if "fair_spread_market" in merged.columns:
        smm = spread_mae(merged["fair_spread_market"], merged["margin"], "fair_spread_market")
        if smm:
            spread_results.append(smm)

    # Calibration table (market prob if available, else model prob)
    if "home_win_prob_market" in merged.columns:
        calib = calibration_table(merged["home_win_prob_market"], merged["home_win"])
        calib_name = "home_win_prob_market"
    elif "home_win_prob" in merged.columns:
        calib = calibration_table(merged["home_win_prob"], merged["home_win"])
        calib_name = "home_win_prob"
    else:
        calib = pd.DataFrame(columns=["bucket", "expected", "actual", "n"])
        calib_name = "none"

    # Write outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUTS_DIR / f"backtest_summary_{start}_{end}.md"
    calib_path = OUTPUTS_DIR / f"backtest_calibration_{start}_{end}.csv"

    lines: List[str] = []
    lines.append(f"# Backtest summary\n")
    lines.append(f"Period: {start} → {end}\n")
    lines.append(f"Games with predictions + results: {len(merged)}\n")

    if brier_results:
        lines.append("\n## Brier scores\n")
        for br in brier_results:
            lines.append(f"- **{br.name}**: {br.brier:.4f} (n={br.n})")
    else:
        lines.append("\n## Brier scores\n- No probability columns found.\n")

    if spread_results:
        lines.append("\n## Spread MAE\n")
        for sm in spread_results:
            lines.append(f"- **{sm.name}**: {sm.mae:.3f} pts (n={sm.n})")
    else:
        lines.append("\n## Spread MAE\n- No fair_spread columns found.\n")

    lines.append(f"\n## Calibration\nUsing: `{calib_name}`\n")
    lines.append(f"- Saved detailed table to `{calib_path.relative_to(REPO_ROOT)}`")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    calib.to_csv(calib_path, index=False)

    print(f"✅ Backtest complete. Summary written to {summary_path}")
    print(f"   Calibration table written to {calib_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Pro-Lite backtesting engine")
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--results",
        default="data/results/results_master.csv",
        help="Path to results CSV (default: data/results/results_master.csv)",
    )

    args = parser.parse_args()
    run_backtest(args.start, args.end, args.results)


if __name__ == "__main__":
    main()
