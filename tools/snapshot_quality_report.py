from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.snapshot_quality import (
    compute_snapshot_quality,
    parse_date_from_filename,
    safe_read_csv,
)

# Conservative defaults (report-only; won’t assume your snapshot schema too aggressively)
DEFAULT_REQUIRED_COLUMNS = [
    "home_team",
    "away_team",
]

# Candidate key triplets (home, away, date). We try these in order.
KEY_TRIPLETS: List[Tuple[str, str, str]] = [
    ("home_team", "away_team", "game_date"),
    ("home_team", "away_team", "date"),
    ("home", "away", "game_date"),
    ("home", "away", "date"),
]

# Market columns commonly present in your prediction outputs and snapshot application
DEFAULT_PROFILE_COLUMNS = [
    "home_ml",
    "away_ml",
    "home_spread",
    "away_spread",
    "spread",
    "total",
    "home_odds",
    "away_odds",
    "line",
    "over_under",
]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _date_in_range(date_str: Optional[str], date_min: Optional[str], date_max: Optional[str]) -> bool:
    if not date_str:
        return True
    if date_min and date_str < date_min:
        return False
    if date_max and date_str > date_max:
        return False
    return True


def _load_predictions_for_date(pred_path: str) -> pd.DataFrame:
    return pd.read_csv(pred_path, dtype=str, keep_default_na=False, na_values=[])


def _build_pred_merge_keys(df_pred: pd.DataFrame) -> pd.Series:
    # Must match locked merge key contract
    return (
        df_pred["home_team"].astype(str).str.lower().str.strip()
        + "__"
        + df_pred["away_team"].astype(str).str.lower().str.strip()
        + "__"
        + df_pred["game_date"].astype(str).str.strip()
    )


def _build_snap_merge_keys(df_snap: pd.DataFrame) -> Optional[pd.Series]:
    cols = set(df_snap.columns)
    # Find a usable home/away/date triplet
    for hc, ac, dc in KEY_TRIPLETS:
        if hc in cols and ac in cols and dc in cols:
            return (
                df_snap[hc].astype(str).str.lower().str.strip()
                + "__"
                + df_snap[ac].astype(str).str.lower().str.strip()
                + "__"
                + df_snap[dc].astype(str).str.strip()
            )
    return None


def _render_md(report: Dict) -> str:
    lines: List[str] = []
    meta = report["meta"]

    lines.append("# Snapshot Quality Report (Audit-Only)")
    lines.append("")
    lines.append(f"- Generated: `{meta['generated_at_utc']}`")
    lines.append(f"- snapshots_dir: `{meta['snapshots_dir']}`")
    lines.append(f"- pred_dir: `{meta.get('pred_dir')}`")
    lines.append(f"- date_min/date_max: `{meta.get('date_min')}` / `{meta.get('date_max')}`")
    lines.append("")

    overall = report["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Snapshot files scanned: **{overall['files_scanned']}**")
    lines.append(f"- Total snapshot rows: **{overall['total_rows']}**")
    lines.append(f"- Files missing required columns: **{overall['files_missing_required_cols']}**")
    lines.append(f"- Files with constant market columns (populated): **{overall['files_with_constant_market_cols']}**")
    if "pred_coverage_dates" in overall:
        lines.append(f"- Dates with prediction coverage computed: **{overall['pred_coverage_dates']}**")
    lines.append("")

    lines.append("## Per-file highlights (worst first)")
    lines.append("")
    lines.append("| Date | File | Rows | Missing req cols | Dup rows | Const market cols | Pred coverage |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for row in report["ranked_files"][:50]:
        lines.append(
            f"| {row.get('date','')} | {Path(row['file']).name} | {row['rows']} | "
            f"{row['missing_required_cols']} | {row['dup_rows']} | {row['constant_market_cols']} | "
            f"{'' if row.get('pred_coverage_pct') is None else str(row['pred_coverage_pct']) + '%'} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Const market cols` counts only columns that are (a) present, (b) mostly populated, and (c) nunique<=1.")
    lines.append("- Prediction coverage compares prediction merge keys to snapshot merge keys for that date, when both exist.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit-only snapshot quality report (no behavior changes).")
    ap.add_argument("--snapshots-dir", required=True, help="Directory with snapshot CSVs.")
    ap.add_argument("--pred-dir", default=None, help="Directory with predictions_YYYY-MM-DD.csv (optional for coverage).")
    ap.add_argument("--date-min", default=None, help="YYYY-MM-DD (optional).")
    ap.add_argument("--date-max", default=None, help="YYYY-MM-DD (optional).")
    ap.add_argument("--out-dir", default="outputs/audits", help="Output directory for report artifacts.")
    ap.add_argument("--required-cols", nargs="*", default=DEFAULT_REQUIRED_COLUMNS, help="Required columns in snapshots.")
    ap.add_argument("--profile-cols", nargs="*", default=DEFAULT_PROFILE_COLUMNS, help="Columns to profile for null/nunique.")
    args = ap.parse_args()

    snap_paths = sorted(glob.glob(str(Path(args.snapshots_dir) / "*.csv")))
    snap_paths = [p for p in snap_paths if _date_in_range(parse_date_from_filename(p), args.date_min, args.date_max)]

    if not snap_paths:
        raise SystemExit("No snapshot CSVs found for given range/dir.")

    # Optional: map date -> prediction file
    pred_paths_by_date: Dict[str, str] = {}
    if args.pred_dir:
        pred_paths = sorted(glob.glob(str(Path(args.pred_dir) / "predictions_*.csv")))
        for p in pred_paths:
            d = parse_date_from_filename(p)
            if d and _date_in_range(d, args.date_min, args.date_max):
                pred_paths_by_date[d] = p

    per_file = []
    overall_rows = 0
    files_missing_req = 0
    files_const_cols = 0
    pred_cov_dates = 0

    for sp in snap_paths:
        df = safe_read_csv(sp)
        q = compute_snapshot_quality(
            df,
            file_path=sp,
            required_columns=list(args.required_cols),
            key_columns_candidates=KEY_TRIPLETS,
            columns_to_profile=list(args.profile_cols),
        )
        overall_rows += q.rows
        if q.missing_required_columns:
            files_missing_req += 1

        const_market_cols = [c for c, is_const in q.constant_flags.items() if is_const]
        if const_market_cols:
            files_const_cols += 1

        # Optional: prediction coverage on same date
        pred_cov_pct = None
        if q.date and q.date in pred_paths_by_date:
            try:
                dfp = _load_predictions_for_date(pred_paths_by_date[q.date])
                if {"home_team", "away_team", "game_date"}.issubset(dfp.columns):
                    pred_keys = set(_build_pred_merge_keys(dfp).tolist())
                    snap_keys_ser = _build_snap_merge_keys(df)
                    if snap_keys_ser is not None:
                        snap_keys = set(snap_keys_ser.tolist())
                        # How many prediction games can be matched to snapshot keys?
                        if pred_keys:
                            cov = 100.0 * (len(pred_keys & snap_keys) / float(len(pred_keys)))
                            pred_cov_pct = round(cov, 2)
                            pred_cov_dates += 1
            except Exception:
                pred_cov_pct = None

        per_file.append(
            {
                "file": q.file,
                "date": q.date,
                "rows": q.rows,
                "cols": q.cols,
                "missing_required_cols": len(q.missing_required_columns),
                "missing_required_columns": q.missing_required_columns,
                "dup_rows": q.duplicate_rows,
                "dup_key_rows": q.duplicate_key_rows,
                "null_rates": q.null_rates,
                "nunique": q.nunique,
                "constant_market_cols": const_market_cols,
                "constant_market_cols_count": len(const_market_cols),
                "pred_coverage_pct": pred_cov_pct,
                "notes": q.notes,
            }
        )

    # Rank worst-first (missing required cols, constant cols, low coverage, dupes, few rows)
    def rank_key(x: Dict) -> tuple:
        cov = 999.0 if x.get("pred_coverage_pct") is None else float(x["pred_coverage_pct"])
        return (
            -int(x["missing_required_cols"]),
            -int(x["constant_market_cols_count"]),
            cov,  # lower is worse
            -int(x["dup_rows"]),
            int(x["rows"]),  # fewer rows is worse but keep deterministic
            str(x.get("date") or ""),
            str(x["file"]),
        )

    ranked = sorted(per_file, key=rank_key)

    report = {
        "meta": {
            "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "snapshots_dir": args.snapshots_dir,
            "pred_dir": args.pred_dir,
            "date_min": args.date_min,
            "date_max": args.date_max,
            "required_cols": list(args.required_cols),
            "profile_cols": list(args.profile_cols),
            "files": snap_paths,
        },
        "overall": {
            "files_scanned": len(snap_paths),
            "total_rows": overall_rows,
            "files_missing_required_cols": files_missing_req,
            "files_with_constant_market_cols": files_const_cols,
            "pred_coverage_dates": pred_cov_dates if args.pred_dir else None,
        },
        "ranked_files": ranked,
        "files": per_file,
    }

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    json_path = out_dir / "snapshot_quality_report.json"
    md_path = out_dir / "snapshot_quality_report.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
