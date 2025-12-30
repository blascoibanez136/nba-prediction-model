"""
Time-of-Day EV Diagnostics (read-only)

Purpose
-------
Quantify how ATS EV and policy eligibility change from OPEN -> CLOSE.

Consumes:
- outputs/backtest_joined.csv (model preds + results)
- data/_snapshots (open/close snapshots; csv and/or json, plus optional raw/ subfolder)
- artifacts/spread_calibrator.joblib
- configs/ats_policy_v1.yaml

Produces:
- outputs/ats_time_of_day_diagnostics.csv   (row-level per game)
- outputs/ats_time_of_day_summary.json      (aggregate summary)

Snapshot support
----------------
OPEN snapshot per day (priority):
  1) open_YYYYMMDD.csv
  2) open_YYYYMMDD*.json
  3) raw_open_YYYY-MM-DD*.json (if present)

CLOSE snapshot per day (priority):
  1) close_YYYYMMDD.csv
  2) close_YYYYMMDD*.json
  3) raw_YYYY-MM-DD*.json (common) / raw_close_YYYY-MM-DD*.json

CSV snapshots must contain:
  - merge_key
  - spread_home_point

JSON snapshots use src.ingest.odds_snapshots.compute_dispersion().
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.ingest.odds_snapshots import compute_dispersion, _norm
from src.model.spread_relative_calibration import load_spread_calibrator, apply_spread_calibrator
from src.utils.policy import load_policy_and_hash


PPU_ATS_MINUS_110 = 100.0 / 110.0  # 0.9090909


# -----------------------------
# helpers
# -----------------------------

def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[tod] Missing date col (expected game_date/date/gamedate).")


def _detect_fair_spread_col(df: pd.DataFrame) -> str:
    for c in ["fair_spread_model", "fair_spread", "fair_spread_close", "spread_pred"]:
        if c in df.columns:
            return c
    raise RuntimeError("[tod] Missing fair spread column (expected fair_spread_model or fair_spread).")


def _candidate_snapshot_dirs(snapshot_dir: Path) -> List[Path]:
    dirs = [snapshot_dir]
    raw = snapshot_dir / "raw"
    if raw.exists() and raw.is_dir():
        dirs.append(raw)
    return dirs


def _pick_first(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p is not None and p.exists() and p.is_file()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[0]


def expected_value_ats(p_win: Optional[float]) -> Optional[float]:
    p = _to_float(p_win)
    if p is None or not (0.0 < p < 1.0):
        return None
    return p * PPU_ATS_MINUS_110 - (1.0 - p)


# -----------------------------
# snapshot discovery
# -----------------------------

def _iter_snapshot_files(snapshot_dir: Path, kind: str, start: str, end: str) -> Iterable[Tuple[str, Path]]:
    """
    Yield (YYYY-MM-DD, path) for one snapshot per day.
    """
    kind = kind.lower().strip()
    if kind not in ("open", "close"):
        raise ValueError("kind must be open|close")

    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    dirs = _candidate_snapshot_dirs(snapshot_dir)

    for d in pd.date_range(start_date, end_date, freq="D"):
        ymd = d.strftime("%Y%m%d")
        ymd_dash = d.strftime("%Y-%m-%d")

        # Prefer CSV snapshots named {kind}_YYYYMMDD.csv
        csv_candidates: List[Path] = []
        for base in dirs:
            csv_candidates.append(base / f"{kind}_{ymd}.csv")
        p = _pick_first(csv_candidates)
        if p:
            yield (ymd_dash, p)
            continue

        # JSON snapshots named {kind}_YYYYMMDD*.json
        json_candidates: List[Path] = []
        for base in dirs:
            json_candidates.extend(list(base.glob(f"{kind}_{ymd}*.json")))
        p = _pick_first(json_candidates)
        if p:
            yield (ymd_dash, p)
            continue

        # raw naming patterns (mostly for CLOSE)
        raw_candidates: List[Path] = []
        for base in dirs:
            if kind == "close":
                raw_candidates.extend(list(base.glob(f"raw_{ymd_dash}*.json")))
                raw_candidates.extend(list(base.glob(f"raw_close_{ymd_dash}*.json")))
            else:
                raw_candidates.extend(list(base.glob(f"raw_open_{ymd_dash}*.json")))
        p = _pick_first(raw_candidates)
        if p:
            yield (ymd_dash, p)
            continue

        # no snapshot for day → skip


# -----------------------------
# compute market stats for a snapshot
# -----------------------------

def _market_from_csv(path: Path) -> pd.DataFrame:
    """
    Compute consensus + dispersion from per-book CSV snapshot.
    Requires columns: merge_key, spread_home_point
    """
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["merge_key", "consensus", "dispersion"])

    if "merge_key" not in df.columns or "spread_home_point" not in df.columns:
        return pd.DataFrame(columns=["merge_key", "consensus", "dispersion"])

    df = df.copy()
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    df["spread_home_point"] = pd.to_numeric(df["spread_home_point"], errors="coerce")
    df = df.dropna(subset=["merge_key", "spread_home_point"])
    if df.empty:
        return pd.DataFrame(columns=["merge_key", "consensus", "dispersion"])

    out = (
        df.groupby("merge_key")["spread_home_point"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "consensus", "std": "dispersion"})
        .reset_index()
    )
    return out


def _market_from_json(path: Path) -> pd.DataFrame:
    """
    Use compute_dispersion JSON helper (returns consensus_close, book_dispersion).
    For OPEN JSON snapshots, the same helper still returns a consensus across books.
    """
    df = compute_dispersion(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["merge_key", "consensus", "dispersion"])

    df = df.rename(columns={"consensus_close": "consensus", "book_dispersion": "dispersion"})
    keep = ["merge_key", "consensus", "dispersion"]
    df = df[keep].copy()
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    return df


def build_market_tables(snapshot_dir: Path, start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      open_market: columns [game_date, merge_key, open_consensus, open_dispersion]
      close_market: columns [game_date, merge_key, close_consensus, close_dispersion]
    """
    open_frames = []
    close_frames = []

    for game_date, path in _iter_snapshot_files(snapshot_dir, "open", start, end):
        m = _market_from_csv(path) if path.suffix.lower() == ".csv" else _market_from_json(path)
        if not m.empty:
            m["game_date"] = game_date
            m = m.rename(columns={"consensus": "open_consensus", "dispersion": "open_dispersion"})
            open_frames.append(m[["game_date", "merge_key", "open_consensus", "open_dispersion"]])

    for game_date, path in _iter_snapshot_files(snapshot_dir, "close", start, end):
        m = _market_from_csv(path) if path.suffix.lower() == ".csv" else _market_from_json(path)
        if not m.empty:
            m["game_date"] = game_date
            m = m.rename(columns={"consensus": "close_consensus", "dispersion": "close_dispersion"})
            close_frames.append(m[["game_date", "merge_key", "close_consensus", "close_dispersion"]])

    open_market = pd.concat(open_frames, ignore_index=True) if open_frames else pd.DataFrame(
        columns=["game_date", "merge_key", "open_consensus", "open_dispersion"]
    )
    close_market = pd.concat(close_frames, ignore_index=True) if close_frames else pd.DataFrame(
        columns=["game_date", "merge_key", "close_consensus", "close_dispersion"]
    )

    # stable + de-dupe per day/merge_key
    if not open_market.empty:
        open_market = open_market.sort_values(["game_date", "merge_key"]).drop_duplicates(["game_date", "merge_key"], keep="last")
    if not close_market.empty:
        close_market = close_market.sort_values(["game_date", "merge_key"]).drop_duplicates(["game_date", "merge_key"], keep="last")

    return open_market, close_market


# -----------------------------
# policy application (read-only)
# -----------------------------

def apply_policy_away_only(
    *,
    ev: pd.Series,
    dispersion: pd.Series,
    max_dispersion: float,
    require_dispersion: bool,
    ev_threshold: float,
) -> pd.Series:
    ev_ok = pd.to_numeric(ev, errors="coerce") >= float(ev_threshold)

    disp = pd.to_numeric(dispersion, errors="coerce")
    if require_dispersion:
        disp_ok = disp.notna() & (disp <= float(max_dispersion))
    else:
        disp_ok = disp.isna() | (disp <= float(max_dispersion))

    return (ev_ok & disp_ok).astype(bool)


# -----------------------------
# main runner
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("time_of_day_ev_diagnostics")
    ap.add_argument("--backtest-joined", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--calibrator", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-csv", default="outputs/ats_time_of_day_diagnostics.csv")
    ap.add_argument("--out-json", default="outputs/ats_time_of_day_summary.json")
    args = ap.parse_args()

    # load policy
    policy_obj, policy_hash = load_policy_and_hash(args.policy)
    ev_threshold = float(policy_obj.get("ev_threshold", 0.03))
    max_dispersion = float(policy_obj.get("max_dispersion", 2.0))
    guards = policy_obj.get("guards", {}) if isinstance(policy_obj.get("guards", {}), dict) else {}
    require_dispersion = True  # ATS policy v1 assumes dispersion gating

    # load backtest joined
    df = pd.read_csv(args.backtest_joined)
    if df.empty:
        raise RuntimeError("[tod] backtest_joined is empty")

    date_col = _detect_date_col(df)
    fair_col = _detect_fair_spread_col(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise RuntimeError("[tod] Found NaT in date column; cannot proceed.")
    df["game_date"] = df[date_col].dt.strftime("%Y-%m-%d")

    # filter range
    df = df[(df["game_date"] >= args.start) & (df["game_date"] <= args.end)].copy()
    if df.empty:
        raise RuntimeError("[tod] no rows in requested date range")

    # merge_key
    if "merge_key" not in df.columns:
        if not {"home_team", "away_team"}.issubset(df.columns):
            raise RuntimeError("[tod] missing merge_key and missing home_team/away_team to construct it.")
        df["merge_key"] = [
            f"{_norm(h)}__{_norm(a)}__{gd}"
            for h, a, gd in zip(df["home_team"], df["away_team"], df["game_date"])
        ]
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()

    # market tables
    open_mkt, close_mkt = build_market_tables(Path(args.snapshot_dir), args.start, args.end)

    # attach
    df = df.merge(open_mkt, on=["game_date", "merge_key"], how="left").merge(
        close_mkt, on=["game_date", "merge_key"], how="left"
    )

    # required: open/close consensus + dispersion
    # If missing, keep as NaN and diagnostic will reflect it.
    df["fair_spread_model"] = pd.to_numeric(df[fair_col], errors="coerce")

    df["residual_open"] = df["fair_spread_model"] - pd.to_numeric(df["open_consensus"], errors="coerce")
    df["residual_close"] = df["fair_spread_model"] - pd.to_numeric(df["close_consensus"], errors="coerce")

    # calibrator
    cal = load_spread_calibrator(args.calibrator)

    # probabilities
    df["p_home_cover_open"] = df.apply(
        lambda r: apply_spread_calibrator(
            residual=_to_float(r["residual_open"]),
            home_spread_consensus=_to_float(r["open_consensus"]),
            calibrator=cal,
        ),
        axis=1,
    )
    df["p_home_cover_close"] = df.apply(
        lambda r: apply_spread_calibrator(
            residual=_to_float(r["residual_close"]),
            home_spread_consensus=_to_float(r["close_consensus"]),
            calibrator=cal,
        ),
        axis=1,
    )

    df["p_away_cover_open"] = 1.0 - pd.to_numeric(df["p_home_cover_open"], errors="coerce")
    df["p_away_cover_close"] = 1.0 - pd.to_numeric(df["p_home_cover_close"], errors="coerce")

    df["ev_away_open"] = df["p_away_cover_open"].apply(expected_value_ats)
    df["ev_away_close"] = df["p_away_cover_close"].apply(expected_value_ats)

    # policy bet flags (away_only)
    df["bet_open"] = apply_policy_away_only(
        ev=df["ev_away_open"],
        dispersion=df["open_dispersion"],
        max_dispersion=max_dispersion,
        require_dispersion=require_dispersion,
        ev_threshold=ev_threshold,
    )
    df["bet_close"] = apply_policy_away_only(
        ev=df["ev_away_close"],
        dispersion=df["close_dispersion"],
        max_dispersion=max_dispersion,
        require_dispersion=require_dispersion,
        ev_threshold=ev_threshold,
    )

    def flip_type(row) -> str:
        bo = bool(row["bet_open"])
        bc = bool(row["bet_close"])
        if bo and bc:
            return "stay_in"
        if (not bo) and (not bc):
            return "stay_out"
        if bo and (not bc):
            return "in_to_out"
        return "out_to_in"

    df["flip_type"] = df.apply(flip_type, axis=1)
    df["ev_delta"] = pd.to_numeric(df["ev_away_close"], errors="coerce") - pd.to_numeric(df["ev_away_open"], errors="coerce")

    # output columns
    out_cols = [
        "game_date",
        "merge_key",
        "home_team",
        "away_team",
        "fair_spread_model",
        "open_consensus",
        "open_dispersion",
        "close_consensus",
        "close_dispersion",
        "residual_open",
        "residual_close",
        "p_away_cover_open",
        "p_away_cover_close",
        "ev_away_open",
        "ev_away_close",
        "bet_open",
        "bet_close",
        "flip_type",
        "ev_delta",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA
    out_df = df[out_cols].copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # summary
    total = int(len(out_df))
    flips = out_df["flip_type"].value_counts(dropna=False).to_dict()
    bet_open_ct = int(out_df["bet_open"].fillna(False).astype(bool).sum())
    bet_close_ct = int(out_df["bet_close"].fillna(False).astype(bool).sum())
    in_to_out = int(flips.get("in_to_out", 0))
    out_to_in = int(flips.get("out_to_in", 0))

    ev_delta = pd.to_numeric(out_df["ev_delta"], errors="coerce")
    summary = {
        "window": {"start": args.start, "end": args.end},
        "policy": {
            "path": args.policy,
            "hash": policy_hash,
            "ev_threshold": ev_threshold,
            "max_dispersion": max_dispersion,
            "require_dispersion": require_dispersion,
        },
        "counts": {
            "rows": total,
            "bet_open": bet_open_ct,
            "bet_close": bet_close_ct,
            "flip_type": flips,
            "open_to_close_retention_rate": (1.0 - (in_to_out / max(bet_open_ct, 1))) if bet_open_ct > 0 else None,
            "close_bets_missing_at_open_rate": (out_to_in / max(bet_close_ct, 1)) if bet_close_ct > 0 else None,
        },
        "ev_delta": {
            "mean": float(ev_delta.mean()) if ev_delta.notna().any() else None,
            "median": float(ev_delta.median()) if ev_delta.notna().any() else None,
            "p10": float(ev_delta.quantile(0.10)) if ev_delta.notna().any() else None,
            "p90": float(ev_delta.quantile(0.90)) if ev_delta.notna().any() else None,
        },
        "dispersion": {
            "open_mean": float(pd.to_numeric(out_df["open_dispersion"], errors="coerce").mean()),
            "close_mean": float(pd.to_numeric(out_df["close_dispersion"], errors="coerce").mean()),
        },
        "artifacts": {
            "diagnostics_csv": str(out_csv),
            "summary_json": str(Path(args.out_json)),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[tod] wrote: {out_csv}")
    print(f"[tod] wrote: {out_json}")
    print(f"[tod] rows={total} bet_open={bet_open_ct} bet_close={bet_close_ct} flips={flips}")


if __name__ == "__main__":
    main()
