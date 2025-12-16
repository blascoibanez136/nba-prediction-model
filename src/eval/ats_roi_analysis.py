"""
ATS (spread) ROI analysis with fixed -110 pricing.

Adds:
- --eval-start / --eval-end date window filtering
- overlap warning if calibrator train window overlaps eval window (metadata-driven)
- --side selection (both/home_only/away_only) for policy testing (default: both)
- --min-abs-residual: residual magnitude gating (default: 0.0 => disabled)

PACKET 1 ADDITIONS (Elite Hardening):
- --policy: load ATS policy YAML (opt-in; preserves legacy CLI behavior if omitted)
- --require-policy-hash: fail-fast if policy hash doesn't match (regression protection)
- policy hash + policy object written into outputs/ats_roi_metrics.json
- best-effort git commit captured for reproducibility

Assumptions:
- pricing is fixed -110 (ppu=0.9090909)
- residual = fair_spread_model - home_spread_consensus (or spread_error fallback)
- calibrator maps residual -> P(home_covers)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.model.spread_relative_calibration import load_spread_calibrator, apply_spread_calibrator
from src.utils.policy import load_policy_and_hash

ATS_ROI_VERSION = "ats_roi_v3_ev_cal_fixed_110_oos_eval_window_residual_gated_2025-12-15"
PPU_ATS_MINUS_110 = 100.0 / 110.0  # 0.9090909


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _to_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _find_score_cols(df: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("home_score", "away_score"),
        ("home_pts", "away_pts"),
        ("home_points", "away_points"),
        ("pts_home", "pts_away"),
    ]
    for h, a in candidates:
        if h in df.columns and a in df.columns:
            return h, a
    raise RuntimeError("[ats_roi] Missing score columns (home_score/away_score or equivalent).")


def _get_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[ats_roi] Missing game_date/date column; required for eval windows.")


def _get_residual(df: pd.DataFrame) -> pd.Series:
    if "fair_spread_model" in df.columns and "home_spread_consensus" in df.columns:
        return pd.to_numeric(df["fair_spread_model"], errors="coerce") - pd.to_numeric(
            df["home_spread_consensus"], errors="coerce"
        )
    if "spread_error" in df.columns:
        return pd.to_numeric(df["spread_error"], errors="coerce")
    raise RuntimeError("[ats_roi] Need fair_spread_model+home_spread_consensus OR spread_error.")


def expected_value_ats(p_win: Optional[float]) -> Optional[float]:
    p = _to_float(p_win)
    if p is None or not (0.0 < p < 1.0):
        return None
    return p * PPU_ATS_MINUS_110 - (1.0 - p) * 1.0


@dataclass(frozen=True)
class ATSConfig:
    per_game_path: str
    calibrator_path: str
    ev_threshold: float = 0.02

    max_dispersion: float = 2.0
    require_dispersion: bool = True

    # residual magnitude gate (abs(residual) >= min_abs_residual)
    min_abs_residual: float = 0.0

    max_bet_rate: float = 0.30
    max_profit_abs: float = 10.0

    eval_start: Optional[str] = None
    eval_end: Optional[str] = None
    fail_on_overlap: bool = False

    # Policy side gating
    side: str = "both"  # "both" | "home_only" | "away_only"

    out_dir: str = "outputs"


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}
    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0).sum())
    roi = (profit / stake) if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean())
    return {"bets": int(len(bets)), "stake": stake, "profit": profit, "roi": roi, "win_rate": win_rate}


def _warn_or_fail_overlap(cal_meta: Dict[str, Any], eval_start: pd.Timestamp, eval_end: pd.Timestamp, fail: bool) -> None:
    ts = cal_meta.get("train_start")
    te = cal_meta.get("train_end")
    if not ts or not te:
        print("[ats] WARNING: calibrator meta missing train_start/train_end; cannot check overlap.")
        return
    t0 = pd.to_datetime(ts, errors="coerce")
    t1 = pd.to_datetime(te, errors="coerce")
    if pd.isna(t0) or pd.isna(t1):
        print("[ats] WARNING: calibrator meta train_start/train_end unparsable; cannot check overlap.")
        return

    overlaps = not (eval_end < t0 or eval_start > t1)
    if overlaps:
        msg = (
            f"[ats] {'ERROR' if fail else 'WARNING'}: calibrator train window "
            f"{t0.date()}..{t1.date()} overlaps eval window {eval_start.date()}..{eval_end.date()} (leakage risk)."
        )
        if fail:
            raise RuntimeError(msg)
        print(msg)
    else:
        print(f"[ats] overlap_check: OK (train {t0.date()}..{t1.date()} vs eval {eval_start.date()}..{eval_end.date()})")


def _apply_side_policy(bets: pd.DataFrame, side: str) -> pd.DataFrame:
    s = str(side).lower().strip()
    if bets is None or bets.empty:
        return bets
    if s == "both":
        return bets
    if s == "home_only":
        return bets[bets["bet_side"].astype(str).str.lower() == "home"].copy()
    if s == "away_only":
        return bets[bets["bet_side"].astype(str).str.lower() == "away"].copy()
    raise ValueError(f"[ats] Invalid side policy: {side}")


def main() -> None:
    ap = argparse.ArgumentParser("ats_roi_analysis.py (OOS eval capable)")
    ap.add_argument("--per_game", required=True)
    ap.add_argument("--calibrator", required=True)
    ap.add_argument("--ev", type=float, default=0.02)

    ap.add_argument("--max-dispersion", type=float, default=2.0)
    ap.add_argument("--no-require-dispersion", action="store_true")

    # residual magnitude gating (disabled by default)
    ap.add_argument(
        "--min-abs-residual",
        type=float,
        default=0.0,
        help="Require abs(spread_residual) >= this value (default: 0.0 = disabled).",
    )

    ap.add_argument("--max-bet-rate", type=float, default=0.30)
    ap.add_argument("--max-profit-abs", type=float, default=10.0)

    # PACKET 1: policy artifact + hash guard (opt-in)
    ap.add_argument(
        "--policy",
        default=None,
        help="Optional path to ATS policy YAML. If provided, parameters will be loaded from it (and hashed/logged).",
    )
    ap.add_argument(
        "--require-policy-hash",
        default=None,
        help="If provided, fail unless computed policy hash matches this value (regression protection).",
    )

    ap.add_argument("--eval-start", default=None, help="Eval window start (YYYY-MM-DD)")
    ap.add_argument("--eval-end", default=None, help="Eval window end inclusive (YYYY-MM-DD)")
    ap.add_argument("--fail-on-overlap", action="store_true", help="Fail loudly if calibrator overlaps eval window")

    ap.add_argument(
        "--side",
        default="both",
        choices=["both", "home_only", "away_only"],
        help="ATS side policy gating (default: both; preserves standard behavior).",
    )

    args = ap.parse_args()

    # PACKET 1: load policy (opt-in)
    policy_obj = None
    policy_path = None
    policy_h = None

    if args.policy:
        policy_path = str(args.policy)
        policy_obj, policy_h = load_policy_and_hash(policy_path)

        # Override CLI args from policy (ONLY when --policy provided)
        if "ev_threshold" in policy_obj:
            args.ev = float(policy_obj["ev_threshold"])
        if "max_dispersion" in policy_obj:
            args.max_dispersion = float(policy_obj["max_dispersion"])
        if "min_abs_residual" in policy_obj:
            args.min_abs_residual = float(policy_obj["min_abs_residual"])
        if "side_policy" in policy_obj:
            args.side = str(policy_obj["side_policy"])

        guards = policy_obj.get("guards", {}) if isinstance(policy_obj.get("guards", {}), dict) else {}
        if "max_bet_rate" in guards:
            args.max_bet_rate = float(guards["max_bet_rate"])
        if "max_profit_abs" in guards:
            args.max_profit_abs = float(guards["max_profit_abs"])

        if args.require_policy_hash:
            if not policy_h:
                raise RuntimeError("[ats] --require-policy-hash provided but no --policy file loaded.")
            if str(args.require_policy_hash).strip() != str(policy_h).strip():
                raise RuntimeError(
                    f"[ats] POLICY HASH MISMATCH: required={args.require_policy_hash} got={policy_h} policy={policy_path}"
                )

    git_commit = _git_commit()

    cfg = ATSConfig(
        per_game_path=args.per_game,
        calibrator_path=args.calibrator,
        ev_threshold=float(args.ev),
        max_dispersion=float(args.max_dispersion),
        require_dispersion=(not bool(args.no_require_dispersion)),
        min_abs_residual=float(args.min_abs_residual),
        max_bet_rate=float(args.max_bet_rate),
        max_profit_abs=float(args.max_profit_abs),
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        fail_on_overlap=bool(args.fail_on_overlap),
        side=str(args.side),
        out_dir="outputs",
    )

    print(f"[ats] version={ATS_ROI_VERSION}")
    print(f"[ats] __file__={__file__}")
    print(f"[ats] cwd={os.getcwd()}")
    print(f"[ats] per_game={cfg.per_game_path}")
    print(f"[ats] calibrator={cfg.calibrator_path}")
    print(f"[ats] ev_threshold={cfg.ev_threshold:.4f}")
    print(f"[ats] min_abs_residual={cfg.min_abs_residual:.3f}")
    print(f"[ats] dispersion<= {cfg.max_dispersion:.2f} require={cfg.require_dispersion}")
    print(f"[ats] guards: max_profit_abs={cfg.max_profit_abs} max_bet_rate={cfg.max_bet_rate}")
    print(f"[ats] pricing: fixed -110 ppu={PPU_ATS_MINUS_110:.6f}")
    print(f"[ats] side_policy={cfg.side}")
    print(f"[ats] git_commit={git_commit}")
    if policy_obj is not None:
        print(
            f"[ats] policy_loaded=True policy_name={policy_obj.get('policy_name')} "
            f"policy_hash={policy_h} policy_path={policy_path}"
        )

    if not os.path.exists(cfg.per_game_path):
        raise FileNotFoundError(f"[ats] per_game not found: {cfg.per_game_path}")
    if not os.path.exists(cfg.calibrator_path):
        raise FileNotFoundError(f"[ats] calibrator not found: {cfg.calibrator_path}")

    df = pd.read_csv(cfg.per_game_path)
    if df.empty:
        raise RuntimeError("[ats] per_game is empty")

    date_col = _get_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise RuntimeError("[ats] Found NaT in game_date; cannot filter safely.")

    # apply eval window filter
    if cfg.eval_start or cfg.eval_end:
        if not (cfg.eval_start and cfg.eval_end):
            raise ValueError("[ats] Provide both --eval-start and --eval-end")
        es = pd.to_datetime(cfg.eval_start, errors="coerce")
        ee = pd.to_datetime(cfg.eval_end, errors="coerce")
        if pd.isna(es) or pd.isna(ee):
            raise ValueError("[ats] Invalid eval window dates.")
        df = df[(df[date_col] >= es) & (df[date_col] <= ee)].copy()
        print(f"[ats] eval_window: {es.date()}..{ee.date()} rows={len(df)}")
    else:
        es = pd.to_datetime(df[date_col].min())
        ee = pd.to_datetime(df[date_col].max())
        print(f"[ats] eval_window: ALL ({es.date()}..{ee.date()}) rows={len(df)}")

    if df.empty:
        raise RuntimeError("[ats] No rows after eval window filtering.")

    if "home_spread_consensus" not in df.columns:
        raise RuntimeError("[ats] Missing required column: home_spread_consensus")

    hcol, acol = _find_score_cols(df)
    df[hcol] = pd.to_numeric(df[hcol], errors="coerce")
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    df["home_spread_consensus"] = pd.to_numeric(df["home_spread_consensus"], errors="coerce")
    df["spread_residual"] = _get_residual(df)

    # dispersion gate
    if "home_spread_dispersion" in df.columns:
        df["home_spread_dispersion"] = pd.to_numeric(df["home_spread_dispersion"], errors="coerce")
        disp_gate = df["home_spread_dispersion"].le(cfg.max_dispersion)
        eligible = (
            disp_gate & df["home_spread_dispersion"].notna()
            if cfg.require_dispersion
            else (disp_gate | df["home_spread_dispersion"].isna())
        )
    else:
        if cfg.require_dispersion:
            raise RuntimeError("[ats] require_dispersion=True but home_spread_dispersion column missing")
        eligible = pd.Series([True] * len(df))

    # residual magnitude gate (abs(residual) >= threshold), only if enabled
    if cfg.min_abs_residual and cfg.min_abs_residual > 0:
        abs_res = pd.to_numeric(df["spread_residual"], errors="coerce").abs()
        eligible = eligible & abs_res.ge(cfg.min_abs_residual)

    eligible = (
        eligible
        & df["spread_residual"].notna()
        & df["home_spread_consensus"].notna()
        & df[hcol].notna()
        & df[acol].notna()
    )
    df["eligible"] = eligible

    cal = load_spread_calibrator(cfg.calibrator_path)
    _warn_or_fail_overlap(cal.get("meta", {}), es, ee, cfg.fail_on_overlap)

    df["p_home_cover"] = df.apply(
        lambda r: apply_spread_calibrator(
            residual=_to_float(r["spread_residual"]),
            home_spread_consensus=_to_float(r["home_spread_consensus"]),
            calibrator=cal,
        ),
        axis=1,
    )
    df["p_away_cover"] = 1.0 - pd.to_numeric(df["p_home_cover"], errors="coerce")
    df["home_ev"] = df["p_home_cover"].apply(expected_value_ats)
    df["away_ev"] = df["p_away_cover"].apply(expected_value_ats)

    def choose_side(r):
        if not bool(r["eligible"]):
            return False, None, None
        hev, aev = r.get("home_ev"), r.get("away_ev")
        if pd.isna(hev) and pd.isna(aev):
            return False, None, None
        if pd.notna(hev) and float(hev) >= cfg.ev_threshold and (pd.isna(aev) or float(hev) > float(aev)):
            return True, "home", float(hev)
        if pd.notna(aev) and float(aev) >= cfg.ev_threshold and (pd.isna(hev) or float(aev) > float(hev)):
            return True, "away", float(aev)
        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0].astype(bool)
    df["bet_side"] = chosen[1]
    df["ev_used"] = chosen[2]

    bets = df[df["bet"]].copy()
    bets = _apply_side_policy(bets, cfg.side)

    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    bet_rate = float(len(bets) / max(total_games, 1))

    if bet_rate > cfg.max_bet_rate:
        raise RuntimeError(
            f"[ats] Bet-rate too high: bets={len(bets)} total_games={total_games} "
            f"bet_rate={bet_rate:.3f} cap={cfg.max_bet_rate}"
        )

    if bets.empty:
        print("[ats] No bets selected.")
        metrics = {
            "version": ATS_ROI_VERSION,
            "bets": 0,
            "bet_rate": bet_rate,
            "side_policy": cfg.side,
            "min_abs_residual": cfg.min_abs_residual,
            "git_commit": git_commit,
            "policy": {
                "loaded": bool(policy_obj is not None),
                "path": policy_path,
                "name": (policy_obj.get("policy_name") if isinstance(policy_obj, dict) else None),
                "version_tag": (policy_obj.get("version_tag") if isinstance(policy_obj, dict) else None),
                "hash": policy_h,
                "object": policy_obj,
            },
        }
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "ats_roi_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        bets.to_csv(os.path.join(cfg.out_dir, "ats_roi_bets.csv"), index=False)
        print(f"[ats] wrote: outputs/ats_roi_metrics.json")
        print(f"[ats] wrote: outputs/ats_roi_bets.csv")
        return

    bets["stake"] = 1.0

    def ats_result(r) -> str:
        hs = float(r[hcol])
        aw = float(r[acol])
        line = float(r["home_spread_consensus"])
        adj_home = hs + line
        if adj_home == aw:
            return "push"
        home_covers = adj_home > aw
        side = str(r["bet_side"]).lower()
        if side == "home":
            return "win" if home_covers else "loss"
        if side == "away":
            return "win" if (not home_covers) else "loss"
        return "unknown"

    bets["result"] = bets.apply(ats_result, axis=1)

    def profit_units(r) -> float:
        res = str(r["result"]).lower()
        if res == "push":
            return 0.0
        if res != "win":
            return -1.0
        return float(PPU_ATS_MINUS_110)

    bets["profit"] = bets.apply(profit_units, axis=1)
    bets["market"] = "spread"

    max_abs = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs > cfg.max_profit_abs:
        raise RuntimeError(f"[ats] Profit sanity failure: max |profit|={max_abs}u limit={cfg.max_profit_abs}")

    overall = summarize(bets)
    home_sum = summarize(bets[bets["bet_side"].astype(str).str.lower() == "home"])
    away_sum = summarize(bets[bets["bet_side"].astype(str).str.lower() == "away"])

    print(f"[ats] overall: {overall}")
    print(f"[ats] home_only: {home_sum}")
    print(f"[ats] away_only: {away_sum}")
    print(f"[ats] bet_rate: {bet_rate:.3f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.out_dir, "ats_roi_metrics.json")
    bets_path = os.path.join(cfg.out_dir, "ats_roi_bets.csv")

    metrics: Dict[str, Any] = {
        "version": ATS_ROI_VERSION,
        "overall": overall,
        "home_only": home_sum,
        "away_only": away_sum,
        "bet_rate": bet_rate,
        "side_policy": cfg.side,
        "ev_threshold": cfg.ev_threshold,
        "min_abs_residual": cfg.min_abs_residual,
        "eval_window": {"start": str(es.date()), "end": str(ee.date()), "date_col": date_col},
        "dispersion": {"max_dispersion": cfg.max_dispersion, "require_dispersion": cfg.require_dispersion},
        "pricing": {"assumed": "-110", "ppu": PPU_ATS_MINUS_110},
        "calibrator_meta": cal.get("meta", {}),
        "git_commit": git_commit,
        "policy": {
            "loaded": bool(policy_obj is not None),
            "path": policy_path,
            "name": (policy_obj.get("policy_name") if isinstance(policy_obj, dict) else None),
            "version_tag": (policy_obj.get("version_tag") if isinstance(policy_obj, dict) else None),
            "hash": policy_h,
            "object": policy_obj,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    bets.to_csv(bets_path, index=False)

    print(f"[ats] wrote: {metrics_path}")
    print(f"[ats] wrote: {bets_path}")


if __name__ == "__main__":
    main()
