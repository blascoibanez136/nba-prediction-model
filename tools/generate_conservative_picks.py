from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.picks.conservative_picks import PicksPolicy, generate_conservative_picks, load_calibration_table


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate conservative picks from prediction CSV (audit-only).")
    ap.add_argument("--pred", required=True, help="Path to predictions_YYYY-MM-DD.csv")
    ap.add_argument("--calibration", default="outputs/backtest_calibration.csv", help="Path to backtest_calibration.csv")
    ap.add_argument("--out-dir", default="outputs", help="Output dir for picks file")
    ap.add_argument("--prob-floor", type=float, default=0.62)
    ap.add_argument("--max-abs-gap", type=float, default=0.08)
    ap.add_argument("--require-cal-keep", action="store_true", default=False)
    ap.add_argument("--max-picks", type=int, default=3)
    ap.add_argument("--min-games", type=int, default=2)

    args = ap.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"Missing pred file: {pred_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = out_dir / "audits"
    audits_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(pred_path)

    calibration_df = None
    cal_path = Path(args.calibration)
    if cal_path.exists():
        calibration_df = load_calibration_table(cal_path)

    policy = PicksPolicy(
        prob_floor=float(args.prob_floor),
        max_abs_gap=float(args.max_abs_gap),
        require_calibration_keep=bool(args.require_cal_keep),
        max_picks_per_day=int(args.max_picks),
        min_games_for_picks=int(args.min_games),
        n_buckets=10,
    )

    picks_df, audit = generate_conservative_picks(preds, calibration_df=calibration_df, policy=policy)

    # Output name: picks_YYYY-MM-DD.csv derived from predictions filename
    stem = pred_path.stem.replace("predictions_", "")
    picks_path = out_dir / f"picks_{stem}.csv"
    picks_df.to_csv(picks_path, index=False)

    audit_path = audits_dir / f"picks_{stem}_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[picks] wrote {picks_path} ({len(picks_df)} rows)")
    print(f"[picks] wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
