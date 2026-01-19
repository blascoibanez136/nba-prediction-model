"""
Totals (Over/Under) ROI analysis with fixed -110 pricing.

PACKET 4A-3:
- Residual magnitude gating
- Optional dispersion gating
- EV × residual interaction (funnel)
- Hard bet-rate + profit guards

PACKET 4B:
- Confidence score + buckets (A/B/C)
- Optional stake sizing:
    - flat (default)
    - bucket (A/B/C stakes)
    - kelly_lite (fractional Kelly with caps)
- Exposure guard on total stake

Hardening tweaks (applied):
1) Drop NaT dates after parsing
2) Coerce fair_total_model + total_consensus to numeric + drop NaNs
3) Clip calibrator probabilities to (1e-6, 1-1e-6)
4) Add --disable-bet-cap flag (default keeps exact v3 behavior)
5) ALWAYS write outputs (metrics + bets CSV), even if 0 bets
6) Coerce home_score/away_score to numeric + drop NaNs
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

from src.model.total_ev import expected_value_total

TOTALS_ROI_VERSION = "totals_roi_v3_buckets_kellylite_2025-12-16_hardened_outputs"
PPU_TOTAL_MINUS_110 = 100.0 / 110.0  # profit per 1u stake at -110


# ---------- helpers ----------

def _get_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date", "gamedate"]:
        if c in df.columns:
            return c
    raise RuntimeError("[totals] Missing game_date/date column")


def summarize(bets: pd.DataFrame) -> Dict[str, Any]:
    if bets is None or bets.empty:
        return {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}

    stake = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(bets["profit"], errors="coerce").fillna(0.0).sum())
    roi = profit / stake if stake > 0 else None
    win_rate = float((bets["result"].astype(str).str.lower() == "win").mean())

    return {
        "bets": int(len(bets)),
        "stake": stake,
        "profit": profit,
        "roi": roi,
        "win_rate": win_rate,
    }


def _parse_bucket_stakes(s: str) -> Dict[str, float]:
    """
    Format: "A=1.0,B=0.6,C=0.3"
    """
    out: Dict[str, float] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().upper()
        v = float(v.strip())
        out[k] = v
    return out


def _assign_buckets_by_quantiles(bets: pd.DataFrame, score_col: str) -> pd.Series:
    """
    A = top 20%
    B = next 30%
    C = remaining 50%
    """
    s = pd.to_numeric(bets[score_col], errors="coerce").astype(float)
    q80 = s.quantile(0.80)
    q50 = s.quantile(0.50)

    def lab(x: float) -> str:
        if x >= q80:
            return "A"
        if x >= q50:
            return "B"
        return "C"

    return s.apply(lab)


def _kelly_fraction(p: float, b: float) -> float:
    """
    Kelly for binary bet with:
      win profit = b (per 1 stake)
      lose loss  = 1 (per 1 stake)
    f* = (b*p - (1-p)) / b
    """
    if p is None or math.isnan(p) or math.isinf(p):
        return 0.0
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser("totals_roi_analysis.py")

    ap.add_argument("--per-game", required=True)
    ap.add_argument("--calibrator", required=True)

    ap.add_argument("--ev", type=float, default=0.04)
    ap.add_argument("--min-abs-residual", type=float, default=2.5)

    
    ap.add_argument(
        "--min-abs-residual-under",
        type=float,
        default=3.5,
        help="Minimum absolute residual for UNDER bets (default 3.5).",
    )
    ap.add_argument("--max-dispersion", type=float, default=8.0)
    ap.add_argument("--require-dispersion", action="store_true")

    ap.add_argument("--side", default="both", choices=["both", "over", "under"])

    ap.add_argument("--eval-start", required=True)
    ap.add_argument("--eval-end", required=True)

    # selection-rate guard (count-based)
    ap.add_argument("--max-bet-rate", type=float, default=0.15)

    # Hardening tweak #4: allow diagnostic runs without trimming
    ap.add_argument(
        "--disable-bet-cap",
        action="store_true",
        help="If set, do NOT trim by --max-bet-rate (diagnostic mode). Default: trimming enabled.",
    )

    # exposure guard (stake-sum-based)
    ap.add_argument(
        "--max-stake-rate",
        type=float,
        default=None,
        help="Cap total stake exposure as (sum stake) / total_games. Default: equals --max-bet-rate.",
    )

    ap.add_argument("--max-profit-abs", type=float, default=10.0)

    # Packet 4B: sizing + buckets
    ap.add_argument(
        "--stake-mode",
        default="flat",
        choices=["flat", "bucket", "kelly_lite"],
        help="flat=1u each, bucket=use A/B/C stakes, kelly_lite=fractional Kelly with caps",
    )
    ap.add_argument("--flat-stake", type=float, default=1.0)
    ap.add_argument("--bucket-stakes", default="A=1.0,B=0.6,C=0.3")
    ap.add_argument("--kelly-fraction", type=float, default=0.25, help="e.g., 0.25 = quarter Kelly")
    ap.add_argument("--kelly-max-unit", type=float, default=1.0)
    ap.add_argument("--kelly-min-unit", type=float, default=0.0)

    ap.add_argument("--out-dir", default="outputs")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bets_path = os.path.join(args.out_dir, "totals_roi_bets.csv")
    metrics_path = os.path.join(args.out_dir, "totals_roi_metrics.json")

    max_stake_rate = args.max_stake_rate if args.max_stake_rate is not None else args.max_bet_rate
    bucket_stakes = _parse_bucket_stakes(args.bucket_stakes) or {"A": 1.0, "B": 0.6, "C": 0.3}

    print(f"[totals] version={TOTALS_ROI_VERSION}")
    print(f"[totals] per_game={args.per_game}")
    print(f"[totals] calibrator={args.calibrator}")
    print(f"[totals] ev_base={args.ev}")
    print(f"[totals] min_abs_residual={args.min_abs_residual}")
    print(f"[totals] side_policy={args.side}")
    print(f"[totals] stake_mode={args.stake_mode}")
    print(f"[totals] max_bet_rate={args.max_bet_rate}")
    print(f"[totals] max_stake_rate={max_stake_rate}")
    print(f"[totals] disable_bet_cap={bool(args.disable_bet_cap)}")

    # -------- initialize metrics shell (so we ALWAYS write) --------
    metrics: Dict[str, Any] = {
        "version": TOTALS_ROI_VERSION,
        "pricing": {"assumed": "-110", "ppu": PPU_TOTAL_MINUS_110},
        "eval_window": {"start": str(args.eval_start), "end": str(args.eval_end)},
        "params": {
            "base_ev": args.ev,
            "min_abs_residual": args.min_abs_residual,
            "min_abs_residual_under": args.min_abs_residual_under,
            "max_dispersion": args.max_dispersion,
            "require_dispersion": bool(args.require_dispersion),
            "max_bet_rate": args.max_bet_rate,
            "disable_bet_cap": bool(args.disable_bet_cap),
            "stake_mode": args.stake_mode,
            "flat_stake": args.flat_stake,
            "bucket_stakes": bucket_stakes,
            "kelly": {
                "fraction": args.kelly_fraction,
                "max_unit": args.kelly_max_unit,
                "min_unit": args.kelly_min_unit,
            },
            "max_stake_rate": max_stake_rate,
            "max_profit_abs": args.max_profit_abs,
            "side_policy": args.side,
        },
        "counts": {},
        "overall": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None},
        "over_only": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None},
        "under_only": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None},
        "buckets": {"A": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None},
                    "B": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None},
                    "C": {"bets": 0, "stake": 0.0, "profit": 0.0, "roi": None, "win_rate": None}},
        "diagnostics": {},
    }

    # ---------- load ----------
    df = pd.read_csv(args.per_game)
    if df.empty:
        raise RuntimeError("[totals] per_game is empty")
    metrics["counts"]["rows_loaded"] = int(len(df))

    date_col = _get_date_col(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Hardening tweak #1: drop NaT dates
    df = df.dropna(subset=[date_col]).copy()
    metrics["counts"]["rows_with_valid_date"] = int(len(df))

    es = pd.to_datetime(args.eval_start, errors="coerce")
    ee = pd.to_datetime(args.eval_end, errors="coerce")
    if pd.isna(es) or pd.isna(ee):
        raise RuntimeError("[totals] invalid eval-start/eval-end date(s)")
    df = df[(df[date_col] >= es) & (df[date_col] <= ee)].copy()
    metrics["counts"]["rows_in_eval_window"] = int(len(df))
    print(f"[totals] eval_window: {es.date()}..{ee.date()} rows={len(df)}")

    if df.empty:
        raise RuntimeError("[totals] No rows in eval window")

    required = ["fair_total_model", "total_consensus", "home_score", "away_score"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"[totals] missing column: {c}")

    # Hardening tweak #2: coerce numerics + drop invalid totals cols
    df["fair_total_model"] = pd.to_numeric(df["fair_total_model"], errors="coerce")
    df["total_consensus"] = pd.to_numeric(df["total_consensus"], errors="coerce")

    # Hardening tweak #6: coerce scores + drop invalid score rows
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=["fair_total_model", "total_consensus", "home_score", "away_score"]).copy()
    metrics["counts"]["rows_after_numeric_coercion"] = int(len(df))
    metrics["diagnostics"]["rows_dropped_numeric"] = int(before_drop - len(df))

    if df.empty:
        raise RuntimeError("[totals] No valid rows after numeric coercion")

    # ---------- core math ----------
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["total_residual"] = df["fair_total_model"] - df["total_consensus"]
    df["abs_residual"] = df["total_residual"].abs()

    # residual magnitude gate
    eligible = df["abs_residual"] >= args.min_abs_residual

    # dispersion gate (optional)
    dispersion_used = False
    if "total_dispersion" in df.columns:
        dispersion_used = True
        df["total_dispersion"] = pd.to_numeric(df["total_dispersion"], errors="coerce")
        disp_ok = df["total_dispersion"] <= args.max_dispersion
        if args.require_dispersion:
            eligible = eligible & disp_ok
        df["gate_dispersion_ok"] = disp_ok
    elif args.require_dispersion:
        raise RuntimeError("[totals] require-dispersion=True but total_dispersion missing")

    df["eligible"] = eligible
    metrics["counts"]["eligible_games"] = int(df["eligible"].sum())
    metrics["diagnostics"]["dispersion_used"] = dispersion_used

    # ---------- calibrator ----------
    cal = joblib.load(args.calibrator)
    iso = cal.get("model") if isinstance(cal, dict) else None
    if iso is None:
        raise RuntimeError("[totals] invalid calibrator artifact (expected dict with key 'model')")

    # Predict P(over) from residual
    df["p_over"] = iso.predict(df["total_residual"].astype(float))

    # Hardening tweak #3: clip to sane probability bounds
    df["p_over"] = pd.to_numeric(df["p_over"], errors="coerce").clip(1e-6, 1.0 - 1e-6)
    df["p_under"] = 1.0 - df["p_over"]

    # EV at -110 (fixed pricing)
    df["over_ev"] = df["p_over"].apply(expected_value_total)
    df["under_ev"] = df["p_under"].apply(expected_value_total)

    # ---------- EV × residual funnel ----------
    def ev_required(abs_res: float) -> float:
        # Larger residual → lower EV requirement (slight)
        if abs_res >= 6.0:
            return max(0.03, args.ev - 0.02)
        if abs_res >= 4.0:
            return max(0.035, args.ev - 0.01)
        return args.ev

    def choose_side(r) -> Tuple[bool, Optional[str], Optional[float]]:
        if not bool(r["eligible"]):
            return False, None, None

        abs_res = float(r["abs_residual"])

        # Asymmetric residual gates:
        # - Over uses --min-abs-residual
        # - Under uses --min-abs-residual-under (default higher)
        res_ok_over = abs_res >= float(args.min_abs_residual)
        res_ok_under = abs_res >= float(args.min_abs_residual_under)

        req_ev = ev_required(abs_res)
        oe, ue = r["over_ev"], r["under_ev"]

        if res_ok_over and pd.notna(oe) and float(oe) >= req_ev and (pd.isna(ue) or float(oe) > float(ue)):
            return True, "over", float(oe)
        if res_ok_under and pd.notna(ue) and float(ue) >= req_ev and (pd.isna(oe) or float(ue) > float(oe)):
            return True, "under", float(ue)
        return False, None, None

    chosen = df.apply(choose_side, axis=1, result_type="expand")
    df["bet"] = chosen[0]
    df["bet_side"] = chosen[1]
    df["ev_used"] = chosen[2]

    bets = df[df["bet"]].copy()
    metrics["counts"]["bets_pre_side_policy"] = int(len(bets))

    # side policy
    if args.side != "both":
        bets = bets[bets["bet_side"] == args.side].copy()
    metrics["counts"]["bets_post_side_policy"] = int(len(bets))

    total_games = int(df["merge_key"].nunique()) if "merge_key" in df.columns else int(len(df))
    metrics["counts"]["total_games"] = int(total_games)

    # ---------- Packet 4A: bet-rate cap via trimming by EV ----------
    cap_applied = False
    cap_max_bets = None
    bets_pre_cap = int(len(bets))

    if not args.disable_bet_cap and not bets.empty:
        cap_applied = True
        cap_max_bets = max(1, int(math.floor(args.max_bet_rate * total_games)))

    if len(bets) > cap_max_bets:
        print(f"[totals] Bet-rate capped: trimming {len(bets)} → {cap_max_bets} (cap={args.max_bet_rate:.2f})")

        # --- side-balanced trimming to avoid pathological one-sided exposure ---
        if "bet_side" in bets.columns and bets["bet_side"].nunique(dropna=True) >= 2:
            cap_each = max(1, cap_max_bets // 2)

            over_bets = bets[bets["bet_side"] == "over"].sort_values("ev_used", ascending=False)
            under_bets = bets[bets["bet_side"] == "under"].sort_values("ev_used", ascending=False)

            kept = pd.concat(
                [over_bets.head(cap_each), under_bets.head(cap_each)],
                ignore_index=True,
            )

            # If one side has fewer than cap_each, backfill remaining slots by EV (regardless of side)
            if len(kept) < cap_max_bets:
                remainder = cap_max_bets - len(kept)
                spill = (
                    bets.sort_values("ev_used", ascending=False)
                    .loc[~bets.index.isin(kept.index)]
                    .head(remainder)
                )
                kept = pd.concat([kept, spill], ignore_index=True)

            bets = kept.sort_values("ev_used", ascending=False).head(cap_max_bets).copy()
        else:
            # Fallback: single-side universe, keep original behavior
            bets = bets.sort_values("ev_used", ascending=False).head(cap_max_bets).copy()

    metrics["diagnostics"]["bet_cap_applied"] = bool(cap_applied)
    metrics["diagnostics"]["cap_max_bets"] = int(cap_max_bets) if cap_max_bets is not None else None
    metrics["counts"]["bets_pre_cap"] = int(bets_pre_cap)
    metrics["counts"]["bets_post_cap"] = int(len(bets))

    # ---------- If no bets: still write outputs ----------
    if bets.empty:
        print("[totals] No bets selected")
        # write empty bets + metrics
        pd.DataFrame().to_csv(bets_path, index=False)
        _write_json(metrics_path, metrics)
        print(f"[totals] wrote {metrics_path}")
        print(f"[totals] wrote {bets_path}")
        return

    # ---------- Packet 4B: confidence score + buckets ----------
    scale = (bets["abs_residual"] / max(args.min_abs_residual, 1e-9)).clip(lower=1.0, upper=2.0)
    bets["confidence_score"] = pd.to_numeric(bets["ev_used"], errors="coerce").astype(float) * scale.astype(float)
    bets["bucket"] = _assign_buckets_by_quantiles(bets, "confidence_score")

    # ---------- Packet 4B: stake sizing ----------
    def stake_for_row(r) -> float:
        if args.stake_mode == "flat":
            return float(args.flat_stake)

        if args.stake_mode == "bucket":
            b = str(r["bucket"]).upper()
            return float(bucket_stakes.get(b, 1.0))

        # kelly_lite
        side = str(r["bet_side"]).lower()
        p = float(r["p_over"]) if side == "over" else float(r["p_under"])
        f = _kelly_fraction(p, PPU_TOTAL_MINUS_110)
        u = args.kelly_fraction * f
        u = max(float(args.kelly_min_unit), min(float(args.kelly_max_unit), float(u)))
        return float(u)

    bets["stake"] = bets.apply(stake_for_row, axis=1)

    # exposure guard: cap SUM(stake) / total_games
    max_total_stake = float(max_stake_rate) * max(total_games, 1)
    if float(bets["stake"].sum()) > max_total_stake:
        bets = bets.sort_values("confidence_score", ascending=False)
        running = bets["stake"].cumsum()
        keep = running <= max_total_stake
        if keep.sum() == 0 and len(bets) > 0:
            keep.iloc[0] = True
        trimmed = int((~keep).sum())
        if trimmed > 0:
            print(f"[totals] Stake-rate capped: trimming {len(bets)} → {int(keep.sum())} (cap={max_stake_rate:.2f})")
        bets = bets.loc[keep].copy()

    bet_rate = float(len(bets)) / max(total_games, 1)
    stake_rate = float(pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0).sum()) / max(total_games, 1)

    # ---------- settlement ----------
    def result(r):
        if float(r["actual_total"]) == float(r["total_consensus"]):
            return "push"
        if str(r["bet_side"]).lower() == "over":
            return "win" if float(r["actual_total"]) > float(r["total_consensus"]) else "loss"
        return "win" if float(r["actual_total"]) < float(r["total_consensus"]) else "loss"

    bets["result"] = bets.apply(result, axis=1)

    def profit(r):
        stake = float(r["stake"])
        if str(r["result"]).lower() == "push":
            return 0.0
        if str(r["result"]).lower() == "win":
            return stake * PPU_TOTAL_MINUS_110
        return -stake

    bets["profit"] = bets.apply(profit, axis=1)

    max_abs_profit = float(pd.to_numeric(bets["profit"], errors="coerce").abs().max())
    if max_abs_profit > float(args.max_profit_abs):
        raise RuntimeError(f"[totals] Profit sanity failure: max |profit|={max_abs_profit} > {args.max_profit_abs}")

    # ---------- outputs ----------
    overall = summarize(bets)
    over_sum = summarize(bets[bets["bet_side"] == "over"])
    under_sum = summarize(bets[bets["bet_side"] == "under"])
    bucket_summaries = {b: summarize(bets[bets["bucket"] == b]) for b in ["A", "B", "C"]}

    # over/under split diagnostics
    side_counts = bets["bet_side"].value_counts(dropna=False).to_dict()
    metrics["diagnostics"]["selected_side_counts"] = side_counts

    metrics["overall"] = overall
    metrics["over_only"] = over_sum
    metrics["under_only"] = under_sum
    metrics["buckets"] = bucket_summaries
    metrics["bet_rate"] = bet_rate
    metrics["stake_rate"] = stake_rate

    print(f"[totals] overall: {overall}")
    print(f"[totals] over_only: {over_sum}")
    print(f"[totals] under_only: {under_sum}")
    print(f"[totals] bet_rate: {bet_rate:.3f}")
    print(f"[totals] stake_rate: {stake_rate:.3f}")
    print(f"[totals] buckets: A={bucket_summaries['A']} B={bucket_summaries['B']} C={bucket_summaries['C']}")

    bets.to_csv(bets_path, index=False)
    _write_json(metrics_path, metrics)

    print(f"[totals] wrote {metrics_path}")
    print(f"[totals] wrote {bets_path}")


if __name__ == "__main__":
    main()
