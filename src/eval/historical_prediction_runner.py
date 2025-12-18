from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "_snapshots"

# Default model search bases (ordered)
DEFAULT_MODEL_BASES = [
    REPO_ROOT / "models",
    REPO_ROOT / "artifacts",
    REPO_ROOT / "artifacts" / "models",
    REPO_ROOT / "outputs",  # sometimes people drop pickles here in Colab
]

AUDIT_PATH = OUTPUTS_DIR / "historical_prediction_runner_audit.json"


@dataclass(frozen=True)
class ModelPaths:
    win_model: Path
    spread_model: Path
    total_model: Path
    team_index: Optional[Path]  # can be None if your predict_games doesn't use it


def _iso(d: date) -> str:
    return d.isoformat()


def _parse_date(s: str) -> date:
    # Accept YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _write_audit(audit: Dict[str, Any]) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def _load_pickle_or_joblib(path: Path) -> Any:
    # Prefer joblib for sklearn artifacts; fallback to pickle.
    import joblib  # type: ignore
    import pickle

    if not path.exists():
        raise FileNotFoundError(str(path))

    # joblib can also load pickles fine in most cases.
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _load_team_index(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env_models_dir() -> Optional[Path]:
    v = os.environ.get("NBA_MODELS_DIR") or os.environ.get("MODELS_DIR")
    if not v:
        return None
    return Path(v).expanduser().resolve()


def _candidate_bases() -> List[Path]:
    bases: List[Path] = []
    env_base = _env_models_dir()
    if env_base:
        bases.append(env_base)
    bases.extend(DEFAULT_MODEL_BASES)
    # de-dupe while preserving order
    seen: set[str] = set()
    out: List[Path] = []
    for b in bases:
        k = str(b)
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out


def _find_first(bases: List[Path], names: List[str], glob_ok: bool = False) -> Optional[Path]:
    """
    Find the first existing file by trying exact names (and optionally globs) across bases.
    """
    for base in bases:
        for name in names:
            p = base / name
            if p.exists():
                return p
        if glob_ok and base.exists():
            for pat in names:
                # allow patterns like "*.pkl"
                for hit in base.glob(pat):
                    if hit.is_file():
                        return hit
    return None


def _locate_model_paths(audit: Dict[str, Any]) -> ModelPaths:
    bases = _candidate_bases()
    audit["model_search"] = {
        "bases": [str(b) for b in bases],
    }

    win = _find_first(bases, ["win_model.pkl", "win_model.joblib"])
    spread = _find_first(bases, ["spread_model.pkl", "spread_model.joblib"])
    total = _find_first(bases, ["total_model.pkl", "total_model.joblib"])

    # team index may be json (your screenshot shows team_index.json)
    team_index = _find_first(bases, ["team_index.json", "teams.json", "teams.pkl", "teams.joblib"])

    missing = [k for k, v in [("win_model", win), ("spread_model", spread), ("total_model", total)] if v is None]
    if missing:
        audit["status"] = "error"
        audit["errors"].append(
            {
                "type": "missing_models",
                "missing": missing,
                "hint": "Expected win_model/spread_model/total_model in models/ or artifacts/. "
                        "You can set NBA_MODELS_DIR to point at the directory.",
            }
        )
        _write_audit(audit)
        raise RuntimeError(f"Could not locate required model artifacts: {', '.join(missing)}. See audit: {AUDIT_PATH}")

    return ModelPaths(
        win_model=win,        # type: ignore[arg-type]
        spread_model=spread,  # type: ignore[arg-type]
        total_model=total,    # type: ignore[arg-type]
        team_index=team_index,
    )


def _load_models(audit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load models into a dict so we can pass them into predict_games in a signature-adaptive way.
    """
    paths = _locate_model_paths(audit)
    audit["model_artifacts"] = {
        "win_model": str(paths.win_model),
        "spread_model": str(paths.spread_model),
        "total_model": str(paths.total_model),
        "team_index": str(paths.team_index) if paths.team_index else None,
    }

    models: Dict[str, Any] = {
        "win_model": _load_pickle_or_joblib(paths.win_model),
        "spread_model": _load_pickle_or_joblib(paths.spread_model),
        "total_model": _load_pickle_or_joblib(paths.total_model),
    }

    # team_index can be json or pickle; load accordingly
    if paths.team_index:
        if paths.team_index.suffix.lower() == ".json":
            models["team_index"] = _load_team_index(paths.team_index)
        else:
            models["team_index"] = _load_pickle_or_joblib(paths.team_index)

    return models


def _read_history_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to normalize expected date column
    if "game_date" not in df.columns:
        # common alternates
        for alt in ["date", "gameDate", "start_date", "startTimeUTC"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "game_date"})
                break
    if "game_date" not in df.columns:
        raise RuntimeError("History CSV must contain a 'game_date' column (or a recognizable alternative).")

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _call_predict_games_signature_adaptive(df_day: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    """
    predict_games signature varies across iterations. We adapt:
    - If it accepts **kwargs like win_model/spread_model/total_model/team_index/teams -> pass them.
    - Otherwise, try common positional layouts.
    """
    from src.model.predict import predict_games  # type: ignore

    sig = inspect.signature(predict_games)
    params = sig.parameters

    # Preferred: pass only supported kwargs
    kwargs: Dict[str, Any] = {}
    for k in ("win_model", "spread_model", "total_model", "team_index", "teams"):
        if k in params and k in models:
            kwargs[k] = models[k]

    # Some versions might accept a single "models" dict
    if "models" in params:
        kwargs["models"] = models

    # Only pass kwargs if they are accepted.
    try:
        out = predict_games(df_day, **kwargs)  # type: ignore[arg-type]
        return out
    except TypeError:
        # Try positional fallbacks in order.
        candidates: List[Tuple[Any, ...]] = []

        # Common: (df, win_model, spread_model, total_model)
        candidates.append((df_day, models["win_model"], models["spread_model"], models["total_model"]))

        # Common: (df, team_index, win_model, spread_model, total_model)
        if "team_index" in models:
            candidates.append((df_day, models["team_index"], models["win_model"], models["spread_model"], models["total_model"]))

        # Common: (df, models_dict)
        candidates.append((df_day, models))

        last_err: Optional[Exception] = None
        for args in candidates:
            try:
                out = predict_games(*args)  # type: ignore[misc]
                return out
            except Exception as e:
                last_err = e

        raise TypeError(
            f"Could not call predict_games with known compatible signatures. "
            f"predict_games signature={sig}. last_error={type(last_err).__name__}: {last_err}"
        )


def _find_close_snapshot_for_date(d: date) -> Optional[Path]:
    """
    Support multiple naming conventions:
    - close_YYYYMMDD.csv  (what your tar.gz extraction shows)
    - odds_snapshot_close_YYYY-MM-DD.csv
    - close_YYYY-MM-DD.csv
    """
    if not SNAPSHOT_DIR.exists():
        return None

    ymd_dash = d.strftime("%Y-%m-%d")
    ymd_compact = d.strftime("%Y%m%d")

    candidates = [
        SNAPSHOT_DIR / f"close_{ymd_compact}.csv",
        SNAPSHOT_DIR / f"close_{ymd_dash}.csv",
        SNAPSHOT_DIR / f"odds_snapshot_close_{ymd_dash}.csv",
        SNAPSHOT_DIR / f"odds_snapshot_close_{ymd_compact}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: any close file containing the date
    pat = re.compile(rf"close_({ymd_compact}|{ymd_dash})\.csv$")
    for p in SNAPSHOT_DIR.glob("close_*.csv"):
        if pat.search(p.name):
            return p
    return None


def _try_apply_market_ensemble(df_pred: pd.DataFrame, d: date) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diag: Dict[str, Any] = {"attempted": True, "applied": False, "reason": None, "snapshot": None}

    snap = _find_close_snapshot_for_date(d)
    if not snap:
        diag["reason"] = "no_close_snapshot"
        return df_pred, diag

    diag["snapshot"] = str(snap)

    try:
        from src.model.market_ensemble import apply_market_ensemble  # type: ignore

        df_out = apply_market_ensemble(df_pred, snapshot_path=str(snap))
        diag["applied"] = True
        return df_out, diag
    except Exception as e:
        diag["reason"] = f"market_apply_error:{type(e).__name__}"
        return df_pred, diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate historical predictions per-day with audits.")
    ap.add_argument("--history", required=True, help="Path to games history CSV (must include game_date).")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--apply-market", action="store_true", help="Attempt market ensemble using close snapshots.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs/predictions_*.csv")
    args = ap.parse_args()

    audit: Dict[str, Any] = {
        "status": "ok",
        "window": {"start": args.start, "end": args.end},
        "inputs": {
            "history": args.history,
            "apply_market": bool(args.apply_market),
            "overwrite": bool(args.overwrite),
        },
        "model_search": {},
        "model_artifacts": {},
        "days": {"processed": 0, "written": 0, "skipped_existing": 0, "skipped_no_games": 0},
        "market": {"attempted_days": 0, "applied_days": 0, "skipped_days": 0, "skip_reasons": {}},
        "outputs_sample": [],
        "errors": [],
    }

    OUTPUTS_DIR.mkdir(exist_ok=True)

    try:
        history_path = Path(args.history)
        if not history_path.exists():
            raise FileNotFoundError(str(history_path))

        df_hist = _read_history_csv(history_path)
        models = _load_models(audit)

        start = _parse_date(args.start)
        end = _parse_date(args.end)

        for d in _date_range(start, end):
            audit["days"]["processed"] += 1
            df_day = df_hist[df_hist["game_date"] == d].copy()
            if df_day.empty:
                audit["days"]["skipped_no_games"] += 1
                continue

            out_path = OUTPUTS_DIR / f"predictions_{_iso(d)}.csv"
            if out_path.exists() and not args.overwrite:
                audit["days"]["skipped_existing"] += 1
                continue

            df_pred = _call_predict_games_signature_adaptive(df_day, models)

            if args.apply_market:
                audit["market"]["attempted_days"] += 1
                df_pred, diag = _try_apply_market_ensemble(df_pred, d)
                if diag.get("applied"):
                    audit["market"]["applied_days"] += 1
                else:
                    audit["market"]["skipped_days"] += 1
                    reason = diag.get("reason") or "unknown"
                    audit["market"]["skip_reasons"][reason] = audit["market"]["skip_reasons"].get(reason, 0) + 1

            df_pred.to_csv(out_path, index=False)
            audit["days"]["written"] += 1

            if len(audit["outputs_sample"]) < 8:
                audit["outputs_sample"].append({"date": _iso(d), "path": str(out_path), "rows": int(len(df_pred))})

        _write_audit(audit)
        print(f"[historical] wrote {AUDIT_PATH}")

    except Exception as e:
        audit["status"] = "error"
        audit["errors"].append({"type": type(e).__name__, "message": str(e)})
        _write_audit(audit)
        raise


if __name__ == "__main__":
    main()
