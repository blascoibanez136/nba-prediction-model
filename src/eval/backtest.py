from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

AUDIT_PATH = OUTPUTS_DIR / "backtest_join_audit.json"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _write_audit(audit: Dict[str, Any]) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def _extract_date_from_pred_filename(name: str) -> Optional[date]:
    """
    Accept:
      predictions_YYYY-MM-DD.csv
      predictions_YYYYMMDD.csv
    """
    m = re.match(r"^predictions_(\d{4}-\d{2}-\d{2})\.csv$", name)
    if m:
        return _parse_date(m.group(1))

    m = re.match(r"^predictions_(\d{8})\.csv$", name)
    if m:
        s = m.group(1)
        return datetime.strptime(s, "%Y%m%d").date()

    return None


def _load_team_index_if_exists() -> Optional[Dict[str, Any]]:
    for p in [
        REPO_ROOT / "models" / "team_index.json",
        REPO_ROOT / "artifacts" / "team_index.json",
    ]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _normalize_team_name(name: str, team_index: Optional[Dict[str, Any]] = None) -> str:
    """
    Minimal canonicalization:
    - lower/strip
    - remove punctuation
    - map a few common NBA variants
    - if team_index exists and has a mapping, prefer it.
    """
    s = (name or "").strip()

    # team_index support (best effort):
    # If team_index includes "name_to_id" or "teams" structures, we can map exact matches.
    if team_index:
        # common patterns we’ve seen in various repos
        for key in ("name_to_id", "teams", "name_map"):
            if key in team_index and isinstance(team_index[key], dict):
                if s in team_index[key]:
                    return s

    x = s.lower()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()

    # common NBA variants
    variants = {
        "la clippers": "los angeles clippers",
        "los angeles clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "los angeles lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        "new york knicks": "new york knicks",
        "okc thunder": "oklahoma city thunder",
        "oklahoma city thunder": "oklahoma city thunder",
        "nop pelicans": "new orleans pelicans",
        "new orleans pelicans": "new orleans pelicans",
    }
    return variants.get(x, x)


def _find_prediction_files(pred_dir: Path, start: date, end: date) -> List[Tuple[date, Path]]:
    if not pred_dir.exists():
        return []

    hits: List[Tuple[date, Path]] = []
    for p in pred_dir.glob("predictions_*.csv"):
        d = _extract_date_from_pred_filename(p.name)
        if not d:
            continue
        if start <= d <= end:
            hits.append((d, p))

    hits.sort(key=lambda t: t[0])
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Join historical prediction CSVs to results history for backtesting.")
    ap.add_argument("--pred-dir", required=True, help="Directory containing predictions_YYYY-MM-DD.csv files.")
    ap.add_argument("--results", required=True, help="History results CSV (e.g., data/history/games_2019_2024.csv).")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "backtest_per_game.csv"), help="Output joined per-game CSV.")
    ap.add_argument("--prob-col", default="home_win_prob_market", help="Which prob column to use (if present).")
    ap.add_argument("--spread-col", default="fair_spread_market", help="Which spread column to use (if present).")
    ap.add_argument("--strict", action="store_true", help="Fail if no prediction files are found in window.")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    results_path = Path(args.results)
    out_path = Path(args.out)

    audit: Dict[str, Any] = {
        "status": "ok",
        "inputs": {
            "pred_dir": str(pred_dir),
            "results": str(results_path),
            "start": args.start,
            "end": args.end,
            "out": str(out_path),
            "prob_col": args.prob_col,
            "spread_col": args.spread_col,
            "strict": bool(args.strict),
        },
        "pred_files": [],
        "counts": {},
        "warnings": [],
        "errors": [],
    }

    try:
        OUTPUTS_DIR.mkdir(exist_ok=True)

        if not results_path.exists():
            raise FileNotFoundError(str(results_path))

        start = _parse_date(args.start)
        end = _parse_date(args.end)

        pred_files = _find_prediction_files(pred_dir, start, end)
        audit["pred_files"] = [{"date": d.isoformat(), "path": str(p)} for d, p in pred_files]

        if not pred_files:
            msg = "No prediction files matched pattern/date window."
            if args.strict:
                audit["status"] = "error"
                audit["errors"].append({"type": "no_predictions", "message": msg})
                _write_audit(audit)
                raise RuntimeError(f"[backtest] {msg} See audit: {AUDIT_PATH}")
            audit["warnings"].append(msg)
            _write_audit(audit)
            print(f"[backtest] wrote {AUDIT_PATH}")
            return

        # Load results
        df_res = pd.read_csv(results_path)
        if "game_date" not in df_res.columns:
            for alt in ["date", "gameDate", "start_date", "startTimeUTC"]:
                if alt in df_res.columns:
                    df_res = df_res.rename(columns={alt: "game_date"})
                    break
        if "game_date" not in df_res.columns:
            raise RuntimeError("Results CSV must contain 'game_date' (or a recognizable alternative).")

        df_res["game_date"] = pd.to_datetime(df_res["game_date"]).dt.date

        # Load & concat prediction files
        dfs: List[pd.DataFrame] = []
        for d, p in pred_files:
            dfp = pd.read_csv(p)
            if "game_date" not in dfp.columns:
                dfp["game_date"] = d
            else:
                dfp["game_date"] = pd.to_datetime(dfp["game_date"]).dt.date
            dfs.append(dfp)

        df_pred = pd.concat(dfs, ignore_index=True)

        # Minimal canonicalization to increase join hit rate
        team_index = _load_team_index_if_exists()

        for col in ["home_team", "away_team"]:
            if col in df_pred.columns:
                df_pred[col] = df_pred[col].astype(str).map(lambda x: _normalize_team_name(x, team_index))
            if col in df_res.columns:
                df_res[col] = df_res[col].astype(str).map(lambda x: _normalize_team_name(x, team_index))

        # Join keys
        required_keys = ["game_date", "home_team", "away_team"]
        for k in required_keys:
            if k not in df_pred.columns:
                raise RuntimeError(f"Predictions missing required column: {k}")
            if k not in df_res.columns:
                raise RuntimeError(f"Results missing required column: {k}")

        df_join = df_pred.merge(df_res, on=required_keys, how="left", suffixes=("", "_res"))

        audit["counts"] = {
            "pred_rows": int(len(df_pred)),
            "res_rows": int(len(df_res)),
            "joined_rows": int(len(df_join)),
            "joined_nonnull_home_score": int(df_join["home_score"].notna().sum()) if "home_score" in df_join.columns else None,
        }

        # Helpful warnings if expected columns aren’t present
        if args.prob_col not in df_join.columns:
            audit["warnings"].append(f"prob_col '{args.prob_col}' not found in joined output.")
        if args.spread_col not in df_join.columns:
            audit["warnings"].append(f"spread_col '{args.spread_col}' not found in joined output.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_join.to_csv(out_path, index=False)

        _write_audit(audit)
        print(f"[backtest] wrote {AUDIT_PATH}")
        print(f"[backtest] wrote {out_path}")

    except Exception as e:
        audit["status"] = "error"
        audit["errors"].append({"type": type(e).__name__, "message": str(e)})
        _write_audit(audit)
        raise


if __name__ == "__main__":
    main()
