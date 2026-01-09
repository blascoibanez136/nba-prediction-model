"""
E5 Regime & Stress Testing
-------------------------

This module implements the E5 regime and stress testing phase for the NBA prediction
system.  It reads the per‑bet decision log produced by the E4 execution policy
(`e4_execution_policy_bets.csv`) and produces diagnostic summaries across
various regimes.  Unlike earlier phases, E5 does not attempt to improve
selection or staking; it exists to measure robustness across environments and to
surface conditions under which betting should pause.

Outputs
=======

The script writes three files into an output directory (default: ``outputs``):

``e5_regime_metrics.json``
    A JSON object containing aggregate metrics sliced by season phase,
    spread magnitude regime and execution window.  Each slice includes
    bet count, ROI (profit per bet), win rate, maximum drawdown and average
    closing line value (CLV) versus the consensus close line.  These are
    descriptive only; they are *not* used to change behaviour automatically.

``e5_kill_switch_diagnostics.json``
    A JSON object recording instances where predefined kill‑switch
    conditions would have triggered during the historical sequence.  Each
    entry includes the date, index and reasons for the trigger (e.g. rolling
    ROI below threshold or drawdown breach).  In E5 this is diagnostic
    only – no automatic halting occurs.

``e5_regime_warnings.csv``
    A simple CSV that flags weekly or period‑level warnings derived from
    the kill‑switch conditions.  It is intended for a human operator to
    review quickly and does not affect the automated pipeline.

Usage
-----

Run this script as a module from the repository root.  For example:

::

    python -m src.eval.e5_regime_audit \
        outputs/e4_execution_policy_bets.csv \
        --out-dir outputs

This command reads the input bets file, computes regime metrics for the
entire season and writes the artefacts into ``outputs``.  You can add
``--start`` and ``--end`` arguments (YYYY‑MM‑DD) to restrict the audit to
sub‑windows (e.g. an out‑of‑sample period).  If these are omitted the
script uses the entire dataset.

Design notes
------------

* The input file must contain at least the following columns:

  - A date column named ``game_date`` or ``date`` or ``gamedate``.  It is
    parsed into a pandas ``datetime64[ns]`` column for slicing and rolling
    computations.
  - A column ``execute_window`` indicating whether the bet was executed
    ``OPEN`` or ``CLOSE``.
  - A numeric column ``executed_home_spread`` giving the spread (home
    perspective) used for bet settlement.  Negative values mean the home
    team was favoured (so away is the underdog).  The absolute value is
    used to assign spread magnitude regimes.
  - Score columns ``home_final_score`` and ``away_final_score`` (or
    ``home_score``/``away_score`` as fallbacks) to determine the actual
    result and compute profit if a ``profit_u`` column is missing.
  - Optionally a ``profit_u`` column giving realised profit in units and a
    ``clv_vs_close`` column measuring the difference between the executed
    spread and the consensus close spread.  If ``profit_u`` is not
    available the script computes it assuming –110 pricing (win = +0.9091
    units, loss = −1.0, push = 0).  If ``clv_vs_close`` is missing the
    script uses NaN for the CLV averages.

* Rolling metrics use window sizes of 20, 30 and 50 bets.  A rolling
  window is defined by count rather than calendar time because bet days
  vary.  Each window computes ROI (mean profit), CLV (mean CLV) and win
  rate.  Drawdown is computed on the cumulative profit series across the
  entire period; the rolling windows only affect ROI/CLV/win rate.

* Kill‑switch diagnostics are based on conservative thresholds:

  - Rolling ROI on a 30‑bet window falling below –0.05 units per bet.
  - Rolling CLV on a 30‑bet window falling below –0.10 points.
  - Cumulative drawdown exceeding –10 units (from the previous peak).

  When any of these occur, the script records the index, date and which
  condition(s) triggered.  Operators can later decide whether these
  thresholds become active stop conditions.

This module intentionally contains no pandas ``SettingWithCopyWarning``
silences.  All operations either create new dataframes or use
``.loc`` assignment carefully.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RegimeMetrics:
    """Data container for aggregated metrics within a regime."""

    bets: int
    roi_per_bet: float
    win_rate: float
    max_drawdown: float
    avg_clv: float


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Infer and parse a date column to datetime.

    The input bets file may contain one of several date column names.
    This helper normalises the date into a ``_dt`` column and returns
    the updated dataframe.

    Raises:
        RuntimeError: if no recognised date column exists or if parsing fails.
    """
    for col in ("game_date", "date", "gamedate"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                df = df.copy()
                df["_dt"] = parsed
                return df
    raise RuntimeError(
        "No recognised date column found; expected one of {game_date,date,gamedate}"
    )


def detect_score_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect the home and away final score columns.

    Returns a tuple (home_col, away_col).  Raises RuntimeError if
    suitable columns cannot be found.
    """
    candidates = [
        ("home_final_score", "away_final_score"),
        ("home_score", "away_score"),
        ("home_pts", "away_pts"),
    ]
    for home_col, away_col in candidates:
        if home_col in df.columns and away_col in df.columns:
            return home_col, away_col
    raise RuntimeError("Cannot find home/away score columns in the input bets file")


def detect_profit_series(df: pd.DataFrame) -> pd.Series:
    """Return the per‑bet profit series as a float numpy array.

    If the input DataFrame contains a ``profit_u`` column, it is
    returned directly.  Otherwise the function uses the result of each
    bet (inferred from the scores and executed spread) and applies
    standard –110 pricing: win = +0.9091 units, loss = −1.0 units,
    push = 0 units.
    """
    if "profit_u" in df.columns:
        return pd.to_numeric(df["profit_u"], errors="coerce").fillna(0.0).to_numpy()
    # Fallback: compute profit from scores and executed spread
    home_col, away_col = detect_score_cols(df)
    spreads = pd.to_numeric(df["executed_home_spread"], errors="coerce").to_numpy()
    home_scores = pd.to_numeric(df[home_col], errors="coerce").to_numpy()
    away_scores = pd.to_numeric(df[away_col], errors="coerce").to_numpy()
    profits: List[float] = []
    ppu = 100.0 / 110.0  # profit per unit when laying -110 odds
    for hs, aw, line in zip(home_scores, away_scores, spreads):
        if np.isnan(hs) or np.isnan(aw) or np.isnan(line):
            profits.append(0.0)
            continue
        adj_home = hs + line  # home spread perspective
        if adj_home == aw:
            profits.append(0.0)  # push
        elif adj_home < aw:
            profits.append(ppu)  # away covers
        else:
            profits.append(-1.0)  # away fails to cover
    return np.array(profits, dtype=float)


def compute_drawdown(series: np.ndarray) -> float:
    """Compute the maximum drawdown (most negative peak‑to‑trough) in a cumulative series."""
    if series.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(series)
    drawdowns = series - running_max
    return float(np.min(drawdowns))


def aggregate_metrics(df: pd.DataFrame) -> RegimeMetrics:
    """Aggregate key metrics for a given slice of the bets DataFrame."""
    n = len(df)
    if n == 0:
        return RegimeMetrics(0, 0.0, 0.0, 0.0, float("nan"))
    profits = detect_profit_series(df)
    roi = float(np.sum(profits) / n)
    win_rate = float(np.mean(profits > 0.0))
    cum = np.cumsum(profits)
    dd = compute_drawdown(cum)
    clv = float(np.nanmean(pd.to_numeric(df.get("clv_vs_close"), errors="coerce")))
    return RegimeMetrics(n, roi, win_rate, dd, clv)


def slice_by_phase(df: pd.DataFrame) -> Dict[str, RegimeMetrics]:
    """Slice the bets by season phase and compute metrics."""
    phases = {}
    # month ranges: early (Oct-Nov), mid (Dec-Jan), post-ASB (Feb-Mar), late (Apr+)
    month = df["_dt"].dt.month
    masks = {
        "early_season": month.isin([10, 11]),
        "mid_season": month.isin([12, 1]),
        "post_all_star": month.isin([2, 3]),
        "late_season": month >= 4,
    }
    for name, mask in masks.items():
        metrics = aggregate_metrics(df[mask])
        phases[name] = metrics
    return phases


def slice_by_spread(df: pd.DataFrame) -> Dict[str, RegimeMetrics]:
    """Slice the bets by absolute executed spread bins and compute metrics."""
    out: Dict[str, RegimeMetrics] = {}
    spreads = pd.to_numeric(df["executed_home_spread"], errors="coerce").abs()
    bins = [0.0, 2.5, 5.0, 8.0, np.inf]
    labels = ["abs_spread_0_2.5", "abs_spread_2.5_5", "abs_spread_5_8", "abs_spread_ge_8"]
    df = df.copy()
    df["spread_bin"] = pd.cut(spreads, bins=bins, labels=labels, include_lowest=True)
    for label in labels:
        metrics = aggregate_metrics(df[df["spread_bin"] == label])
        out[str(label)] = metrics
    # additional sign category: away favourite (home dog)
    away_fav_mask = pd.to_numeric(df["executed_home_spread"], errors="coerce") > 0
    out["away_favourite"] = aggregate_metrics(df[away_fav_mask])
    out["away_underdog"] = aggregate_metrics(df[~away_fav_mask])
    return out


def slice_by_execution_window(df: pd.DataFrame) -> Dict[str, RegimeMetrics]:
    """Slice the bets by execution window (OPEN/CLOSE) and compute metrics."""
    out: Dict[str, RegimeMetrics] = {}
    for window in ["OPEN", "CLOSE"]:
        metrics = aggregate_metrics(df[df["execute_window"] == window])
        out[window] = metrics
    return out


def rolling_metrics(df: pd.DataFrame, profits: np.ndarray, clv: np.ndarray, windows: List[int]) -> Dict[int, Dict[str, List[float]]]:
    """Compute rolling ROI, CLV and win rate over specified window sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Bets dataframe sorted by date.
    profits : np.ndarray
        Profit per bet array.
    clv : np.ndarray
        CLV per bet array (NaN if not available).
    windows : list
        List of integer window sizes.

    Returns
    -------
    dict
        Keys are window sizes; values are dictionaries with lists of
        ``roi``, ``clv`` and ``win_rate`` aligned with the original
        index (the first ``window-1`` entries will be NaN).
    """
    n = len(df)
    result: Dict[int, Dict[str, List[float]]] = {}
    for w in windows:
        roi_series: List[float] = [float("nan")] * n
        clv_series: List[float] = [float("nan")] * n
        win_series: List[float] = [float("nan")] * n
        if w <= 0:
            continue
        for i in range(w - 1, n):
            window_profits = profits[i - w + 1 : i + 1]
            roi_series[i] = float(np.mean(window_profits))
            if clv is not None and len(clv) == n:
                window_clv = clv[i - w + 1 : i + 1]
                clv_series[i] = float(np.nanmean(window_clv))
            win_series[i] = float(np.mean(window_profits > 0.0))
        result[w] = {"roi": roi_series, "clv": clv_series, "win_rate": win_series}
    return result


def evaluate_kill_switch(df: pd.DataFrame, profits: np.ndarray, clv: np.ndarray) -> List[Dict]:
    """Evaluate kill‑switch conditions and record trigger points.

    Returns a list of dictionaries with keys: index, date, and reasons.
    """
    triggers: List[Dict] = []
    n = len(df)
    # compute cumulative profit and drawdown to evaluate drawdown trigger
    cum = np.cumsum(profits)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    # compute rolling 30‑bet ROI and CLV
    w = 30
    roi_roll = [float("nan")] * n
    clv_roll = [float("nan")] * n
    for i in range(w - 1, n):
        window_profits = profits[i - w + 1 : i + 1]
        roi_roll[i] = float(np.mean(window_profits))
        if clv is not None and len(clv) == n:
            window_clv = clv[i - w + 1 : i + 1]
            clv_roll[i] = float(np.nanmean(window_clv))
    for i in range(n):
        reasons: List[str] = []
        if roi_roll[i] is not None and not np.isnan(roi_roll[i]) and roi_roll[i] < -0.05:
            reasons.append("rolling_roi_below_-0.05")
        if clv_roll[i] is not None and not np.isnan(clv_roll[i]) and clv_roll[i] < -0.10:
            reasons.append("rolling_clv_below_-0.10")
        if drawdown[i] < -10.0:
            reasons.append("drawdown_below_-10u")
        if reasons:
            triggers.append({
                "index": int(i),
                "date": str(df.iloc[i]["_dt"].date()),
                "reasons": reasons,
            })
    return triggers


def write_outputs(
    out_dir: Path,
    metrics: Dict[str, Dict[str, RegimeMetrics]],
    kill_switches: List[Dict],
    warnings: pd.DataFrame,
) -> None:
    """Write JSON and CSV artefacts to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # serialise metrics
    serialised: Dict[str, Dict[str, Dict[str, float]]] = {}
    for category, slices in metrics.items():
        serialised[category] = {}
        for key, metr in slices.items():
            serialised[category][key] = asdict(metr)
    (out_dir / "e5_regime_metrics.json").write_text(
        json.dumps(serialised, indent=2), encoding="utf-8"
    )
    # kill switches
    (out_dir / "e5_kill_switch_diagnostics.json").write_text(
        json.dumps(kill_switches, indent=2), encoding="utf-8"
    )
    # warnings
    warnings.to_csv(out_dir / "e5_regime_warnings.csv", index=False)


def compute_warnings(
    df: pd.DataFrame,
    triggers: List[Dict],
    window_days: int = 7,
) -> pd.DataFrame:
    """Generate high‑level regime warnings grouped by calendar week.

    This function groups the kill‑switch triggers by week and constructs
    summary flags that a human operator can inspect quickly.  It also
    reports periods where the proportion of close bets exceeds a threshold.
    """
    if df.empty:
        return pd.DataFrame()
    # convert _dt to date for grouping by week number
    df = df.copy()
    df["date_only"] = df["_dt"].dt.date
    df["week"] = df["_dt"].dt.isocalendar().week
    df["year"] = df["_dt"].dt.isocalendar().year
    # prepare kill switch map
    trigger_df = pd.DataFrame(triggers)
    if not trigger_df.empty:
        trigger_df["week"] = pd.to_datetime(trigger_df["date"]).dt.isocalendar().week
        trigger_df["year"] = pd.to_datetime(trigger_df["date"]).dt.isocalendar().year
    else:
        trigger_df = pd.DataFrame(columns=["week", "year", "reasons"])
    # compute weekly summary
    summary_rows: List[Dict] = []
    grouped = df.groupby(["year", "week"])
    for (yr, wk), group in grouped:
        row: Dict[str, object] = {
            "year": int(yr),
            "week": int(wk),
            "bets": int(len(group)),
        }
        # count triggers in this week
        triggers_in_week = trigger_df[(trigger_df["year"] == yr) & (trigger_df["week"] == wk)]
        row["trigger_count"] = int(len(triggers_in_week))
        # list unique trigger reasons
        if not triggers_in_week.empty:
            reasons_flat = [r for reasons in triggers_in_week["reasons"] for r in reasons]
            row["trigger_reasons"] = ";".join(sorted(set(reasons_flat)))
        else:
            row["trigger_reasons"] = ""
        # proportion of CLOSE bets
        close_prop = float(np.mean(group["execute_window"] == "CLOSE")) if len(group) > 0 else 0.0
        row["close_bet_share"] = close_prop
        # flag high close share > 0.5 (arbitrary diagnostic)
        row["high_close_share_warning"] = bool(close_prop > 0.5)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def run_regime_audit(bets_path: str, out_dir: str, start: Optional[str], end: Optional[str]) -> None:
    df = pd.read_csv(bets_path)
    if df.empty:
        raise RuntimeError("Input bets CSV is empty")
    # parse date
    df = parse_dates(df)
    # filter by start/end
    if start:
        df = df[df["_dt"] >= pd.to_datetime(start)]
    if end:
        df = df[df["_dt"] <= pd.to_datetime(end)]
    df = df.sort_values("_dt").reset_index(drop=True)
    # ensure required columns exist
    required_cols = ["execute_window", "executed_home_spread"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Required column '{col}' missing from bets file")
    # compute profits and clv arrays
    profits = detect_profit_series(df)
    clv_arr = None
    if "clv_vs_close" in df.columns:
        clv_arr = pd.to_numeric(df["clv_vs_close"], errors="coerce").to_numpy()
    # compute metrics
    metrics: Dict[str, Dict[str, RegimeMetrics]] = {
        "season_phase": slice_by_phase(df),
        "spread_regime": slice_by_spread(df),
        "execution_window": slice_by_execution_window(df),
    }
    # rolling metrics (not part of output, but could be saved separately if desired)
    roll = rolling_metrics(df, profits, clv_arr, windows=[20, 30, 50])
    # evaluate kill switches
    triggers = evaluate_kill_switch(df, profits, clv_arr if clv_arr is not None else np.full(len(profits), np.nan))
    warnings_df = compute_warnings(df, triggers)
    write_outputs(Path(out_dir), metrics, triggers, warnings_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 Regime audit and stress diagnostics")
    parser.add_argument(
        "bets_csv", type=str, help="Path to the E4 execution policy bets CSV file"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory to write E5 artefacts (defaults to 'outputs')",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start date (YYYY-MM-DD) for audit"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End date (YYYY-MM-DD) for audit"
    )
    args = parser.parse_args()
    run_regime_audit(args.bets_csv, args.out_dir, args.start, args.end)


if __name__ == "__main__":
    main()
