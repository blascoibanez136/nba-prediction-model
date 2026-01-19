# src/eval/build_totals_roi_input.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def _norm(s: object) -> str:
    return str(s).strip().lower()


def _candidate_snapshot_dirs(snapshot_dir: Path) -> List[Path]:
    dirs = [snapshot_dir]
    raw = snapshot_dir / "raw"
    if raw.exists() and raw.is_dir():
        dirs.append(raw)
    return dirs


def _pick_first(paths: List[Path]) -> Optional[Path]:
    paths = [p for p in paths if p and p.exists() and p.is_file()]
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.name)[0]


def _iter_close_snapshot_files(snapshot_dir: Path, start: str, end: str) -> Iterable[Path]:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    dirs = _candidate_snapshot_dirs(snapshot_dir)

    for d in pd.date_range(start_date, end_date, freq="D"):
        ymd = d.strftime("%Y%m%d")
        ymd_dash = d.strftime("%Y-%m-%d")

        csv_candidates: List[Path] = []
        for base in dirs:
            csv_candidates.append(base / f"close_{ymd}.csv")
        p = _pick_first(csv_candidates)
        if p:
            yield p
            continue

        json_candidates: List[Path] = []
        for base in dirs:
            json_candidates.extend(list(base.glob(f"close_{ymd}*.json")))
        p = _pick_first(json_candidates)
        if p:
            yield p
            continue

        raw_candidates: List[Path] = []
        for base in dirs:
            raw_candidates.extend(list(base.glob(f"raw_{ymd_dash}*.json")))
            raw_candidates.extend(list(base.glob(f"raw_close_{ymd_dash}*.json")))
        p = _pick_first(raw_candidates)
        if p:
            yield p
            continue


def _market_from_close_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty or "merge_key" not in df.columns:
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    total_col = None
    for c in ["total_point", "total", "total_points", "points_total"]:
        if c in df.columns:
            total_col = c
            break
    if total_col is None:
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    df = df.copy()
    df["merge_key"] = df["merge_key"].astype(str).str.strip().str.lower()
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df = df.dropna(subset=["merge_key", total_col])
    if df.empty:
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    out = (
        df.groupby("merge_key")[total_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "total_consensus", "std": "total_dispersion"})
        .reset_index()
    )
    return out


def _market_from_oddsapi_json(path: Path) -> pd.DataFrame:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    rows = []
    if not isinstance(data, list):
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    for game in data:
        mk = game.get("merge_key")
        if not mk:
            continue
        books = game.get("bookmakers", [])
        for b in books:
            for m in b.get("markets", []):
                key = m.get("key") or m.get("market_key")
                if str(key).lower() != "totals":
                    continue
                for o in m.get("outcomes", []):
                    pt = o.get("point")
                    if pt is None:
                        continue
                    rows.append({"merge_key": _norm(mk), "total_point": float(pt)})

    if not rows:
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    df = pd.DataFrame(rows)
    out = (
        df.groupby("merge_key")["total_point"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "total_consensus", "std": "total_dispersion"})
        .reset_index()
    )
    return out


def build_market_df(snapshot_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for p in _iter_close_snapshot_files(snapshot_dir, start, end):
        try:
            if p.suffix.lower() == ".csv":
                frames.append(_market_from_close_csv(p))
            else:
                frames.append(_market_from_oddsapi_json(p))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["merge_key", "total_consensus", "total_dispersion"])

    market = pd.concat(frames, ignore_index=True)
    market["merge_key"] = market["merge_key"].astype(str).str.strip().str.lower()
    market = market.sort_values("merge_key").drop_duplicates("merge_key", keep="last")
    return market


def _ensure_merge_key(back: pd.DataFrame) -> pd.DataFrame:
    # If already present, normalize it.
    if "merge_key" in back.columns:
        back = back.copy()
        back["merge_key"] = back["merge_key"].astype(str).str.strip().str.lower()
        return back

    # Otherwise construct from home_team / away_team / game_date|date
    if not {"home_team", "away_team"}.issubset(back.columns):
        raise RuntimeError("[build_totals_roi_input] missing merge_key and missing home_team/away_team")

    date_col = None
    for c in ["game_date", "date", "gamedate"]:
        if c in back.columns:
            date_col = c
            break
    if date_col is None:
        raise RuntimeError("[build_totals_roi_input] missing merge_key and missing date column")

    dt = pd.to_datetime(back[date_col], errors="coerce")
    if dt.isna().any():
        raise RuntimeError("[build_totals_roi_input] Found NaT in date column; cannot build merge_key safely.")
    date_str = dt.dt.strftime("%Y-%m-%d")

    back = back.copy()
    back["merge_key"] = [
        f"{_norm(h)}__{_norm(a)}__{d}"
        for h, a, d in zip(back["home_team"], back["away_team"], date_str)
    ]
    back["merge_key"] = back["merge_key"].astype(str).str.strip().str.lower()
    return back


def main() -> None:
    ap = argparse.ArgumentParser("build_totals_roi_input")
    ap.add_argument("--backtest-joined", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    back = pd.read_csv(args.backtest_joined)
    if back.empty:
        raise RuntimeError("[build_totals_roi_input] backtest_joined is empty")

    back = _ensure_merge_key(back)

    # Rename fair_total -> fair_total_model if needed
    if "fair_total_model" not in back.columns and "fair_total" in back.columns:
        back = back.rename(columns={"fair_total": "fair_total_model"})

    market = build_market_df(Path(args.snapshot_dir), args.start, args.end)
    out_df = back.merge(market, on="merge_key", how="left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    cov = float(out_df["total_consensus"].notna().mean() * 100.0) if "total_consensus" in out_df.columns else 0.0
    print(f"[build_totals_roi_input] wrote: {args.out} (total_consensus_cov={cov:.1f}%)")


if __name__ == "__main__":
    main()
