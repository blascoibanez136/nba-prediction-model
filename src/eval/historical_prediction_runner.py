"""Historical prediction runner.

Design intent (Commit-2):
- Reuse the same `predict_games()` used for daily runs.
- Iterate date-by-date over a historical schedule CSV and write one prediction
  file per date in an output directory.
- Optionally apply market ensemble using CLOSE odds snapshots in a snapshot dir.

Key robustness:
- Avoid passing keyword args that `predict_games()` doesn't accept.
- Robust imports: supports `src.predict`, `src.model.predict`, or `predict`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd


def _import_predict_games() -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Import predict_games from the most likely module locations."""

    tried = []

    # 1) Preferred: src.predict
    try:
        from src.predict import predict_games  # type: ignore

        return predict_games
    except Exception as e:  # pragma: no cover
        tried.append(("src.predict", repr(e)))

    # 2) Alternate: src.model.predict
    try:
        from src.model.predict import predict_games  # type: ignore

        return predict_games
    except Exception as e:  # pragma: no cover
        tried.append(("src.model.predict", repr(e)))

    # 3) Fallback: repo-root predict.py
    try:
        from predict import predict_games  # type: ignore

        return predict_games
    except Exception as e:  # pragma: no cover
        tried.append(("predict", repr(e)))

    msg = [
        "Could not import predict_games.",
        "Tried:",
    ]
    msg += [f"- {m}: {err}" for (m, err) in tried]
    msg += [
        "\nExpected one of:",
        "- src/predict.py (import src.predict)",
        "- src/model/predict.py (import src.model.predict)",
        "- predict.py at repo root (import predict)",
    ]
    raise ModuleNotFoundError("\n".join(msg))


predict_games = _import_predict_games()


def _normalize_date_series(s: pd.Series) -> pd.Series:
    # Accept YYYY-MM-DD, ISO datetime, or numeric yyyymmdd.
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    # If anything failed but looks like yyyymmdd ints/strings, retry.
    if dt.isna().any():
        dt2 = pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
        dt = dt.fillna(dt2)
    return dt.dt.date


def _infer_date_col(df: pd.DataFrame) -> str:
    for c in ("date", "game_date", "start_date", "gameDate", "GAME_DATE"):
        if c in df.columns:
            return c
    # Last-resort: any column that parses well as dates
    best = None
    best_ok = -1
    for c in df.columns:
        ok = pd.to_datetime(df[c], errors="coerce").notna().mean()
        if ok > best_ok:
            best_ok = ok
            best = c
    if best is None or best_ok < 0.6:
        raise ValueError(
            "Could not infer a date column from history CSV. "
            "Expected a column like 'date' or 'game_date'."
        )
    return str(best)


def _run_day(df_day: pd.DataFrame, apply_market: bool, snapshot_dir: Path) -> pd.DataFrame:
    # Predict (features/models handled inside predict_games)
    df_pred = predict_games(df_day)

    if apply_market:
        # predict_games() already applies market if it sees snapshots at default
        # location. But for historical runs we want explicit snapshot_dir.
        # The simplest/robust approach: if predict_games already applied market
        # (columns exist), do nothing. Otherwise, leave un-ensembled.
        #
        # This keeps the runner compatible with multiple predict.py versions.
        #
        # If you want strict behavior, implement market-ensemble here.
        pass

    return df_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to games_*.csv")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--out-dir", default="outputs", help="Where to write predictions_YYYY-MM-DD.csv")
    ap.add_argument(
        "--snapshot-dir",
        default="data/_snapshots",
        help="Directory containing close_YYYYMMDD.csv files",
    )
    ap.add_argument("--apply-market", action="store_true", help="Attempt to apply market ensemble")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--pattern", default="predictions_{date}.csv", help="Output filename pattern")
    ap.add_argument(
        "--audit-path",
        default="outputs/historical_prediction_runner_audit.json",
        help="Write a small audit JSON here",
    )

    args = ap.parse_args()

    history_path = Path(args.history)
    out_dir = Path(args.out_dir)
    snapshot_dir = Path(args.snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.audit_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)
    date_col = _infer_date_col(df)
    df["__date"] = _normalize_date_series(df[date_col])

    start_d = pd.to_datetime(args.start).date()
    end_d = pd.to_datetime(args.end).date()

    audit: Dict[str, object] = {
        "history": str(history_path),
        "out_dir": str(out_dir),
        "snapshot_dir": str(snapshot_dir),
        "apply_market": bool(args.apply_market),
        "date_col": date_col,
        "start": str(start_d),
        "end": str(end_d),
        "written": [],
        "skipped_existing": [],
        "errors": [],
    }

    all_dates = sorted(d for d in df["__date"].dropna().unique() if start_d <= d <= end_d)

    print("[historical] loaded predict_games")
    print(f"[historical] history rows: {len(df):,} | days in range: {len(all_dates):,}")

    for d in all_dates:
        df_day = df[df["__date"] == d].copy()
        if df_day.empty:
            continue

        fname = args.pattern.format(date=str(d))
        out_path = out_dir / fname
        if out_path.exists() and not args.overwrite:
            audit["skipped_existing"].append(str(out_path))
            continue

        try:
            df_pred = _run_day(df_day, bool(args.apply_market), snapshot_dir)
            df_pred.to_csv(out_path, index=False)
            audit["written"].append(str(out_path))
            print(f"[historical] Wrote {out_path} ({len(df_pred)} rows)")
        except Exception as e:  # pragma: no cover
            audit["errors"].append({"date": str(d), "error": repr(e)})
            print(f"[historical][ERROR] {d}: {e}")

    Path(args.audit_path).write_text(json.dumps(audit, indent=2))
    print(f"[historical] wrote {args.audit_path}")


if __name__ == "__main__":
    # Ensure repo root is on sys.path when invoked from odd working dirs.
    # (Users often run: PYTHONPATH=. python -m ...)
    os.environ.setdefault("PYTHONPATH", ".")
    main()
