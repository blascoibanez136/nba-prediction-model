# qa/regression_totals_policy_v3.py
import json
from pathlib import Path


def main():
    metrics_path = Path("outputs/totals_roi_metrics.json")
    if not metrics_path.exists():
        raise RuntimeError("[totals_regression] Missing outputs/totals_roi_metrics.json")

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    overall = m.get("overall", {})
    bets = overall.get("bets", 0)
    roi = overall.get("roi", None)

    # Shadow sanity checks (initially conservative)
    # Adjust once you see real totals performance.
    if bets is None or int(bets) < 10:
        raise RuntimeError(f"[totals_regression] Too few totals bets: bets={bets} (shadow expects >=10)")

    if roi is None:
        raise RuntimeError("[totals_regression] Missing ROI in totals metrics")

    if float(roi) < -0.05:
        raise RuntimeError(f"[totals_regression] Totals ROI too negative: roi={roi}")

    print("[totals_regression] ✅ totals shadow regression passed")
    print(f"[totals_regression] bets={bets} roi={float(roi):.4f}")


if __name__ == "__main__":
    main()
