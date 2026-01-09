#!/usr/bin/env python
"""
Command-line entrypoint for running the E6 portfolio overlay.

This script loads a portfolio policy, reads candidate bet data (preferably
pre-staked by E3), applies the E6 portfolio overlay using the functions in
`e6_portfolio.py`, and writes portfolio outputs.

Usage:
  python e6_policy_runner.py \
    --policy configs/e6_portfolio_policy_v1.json \
    --candidates outputs/ats_e3_staked_bets.csv

Outputs (as defined in the policy file) are:
  - bets_csv
  - metrics_json
  - audit_json
"""

import argparse
import json
from pathlib import Path
import pandas as pd

# Import portfolio functions from the src package. When running from repo root,
# ensure PYTHONPATH=. so that `src` is in the search path.
try:
    from src.portfolio.e6_portfolio import load_policy, apply_e6_portfolio, compute_metrics
except ModuleNotFoundError:
    # Fallback for local execution when files are colocated in the same directory
    from e6_portfolio import load_policy, apply_e6_portfolio, compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Run E6 portfolio overlay')
    parser.add_argument('--policy', default='configs/e6_portfolio_policy_v1.json', help='Path to E6 portfolio policy JSON')
    parser.add_argument('--candidates', default='outputs/ats_e3_staked_bets.csv', help='Path to candidate bets CSV (post-E3 preferred)')
    args = parser.parse_args()

    # Load policy
    policy = load_policy(args.policy)

    # Load candidates
    candidate_path = Path(args.candidates)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate bets file not found: {candidate_path}")
    df = pd.read_csv(candidate_path)

    # Harmonize date column if necessary
    if 'game_date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'game_date'})

    # Apply E6 portfolio overlay (includes caps and kill-switch logic)
    portfolio_df = apply_e6_portfolio(df, policy)

    # Compute simple metrics
    metrics = compute_metrics(portfolio_df, policy)

    # Determine output paths from policy
    outputs = policy.get('interfaces', {}).get('outputs', {})
    bets_path = Path(outputs.get('bets_csv', 'outputs/e6_portfolio_bets.csv'))
    metrics_path = Path(outputs.get('metrics_json', 'outputs/e6_portfolio_metrics.json'))
    audit_path = Path(outputs.get('audit_json', 'outputs/e6_portfolio_audit.json'))

    # Ensure directories exist
    bets_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    # Write portfolio bets CSV
    portfolio_df.to_csv(bets_path, index=False)

    # Write metrics JSON
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Write audit JSON with reason breakdown
    # Count accepted and rejected bets and reasons
    reason_counts = portfolio_df['e6_reason'].value_counts().to_dict() if 'e6_reason' in portfolio_df.columns else {}
    selected_count = int((portfolio_df.get('stake_u_final', 0) > 0).sum())
    rejected_count = int((portfolio_df.get('stake_u_final', 0) <= 0).sum())
    audit = {
        'selected': selected_count,
        'rejected': rejected_count,
        'reasons': reason_counts,
        'note': 'E6 overlay with caps, drawdown brakes and kill-switch applied'
    }
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)

    print(f"Portfolio run complete. Bets: {len(portfolio_df)}. Outputs written to {bets_path}, {metrics_path}, {audit_path}.")


if __name__ == '__main__':
    main()
