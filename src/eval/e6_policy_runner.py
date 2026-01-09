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

    # Apply E6 portfolio overlay (skeleton)
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

    # Write audit JSON (placeholder)
    audit = {
        'selected': int(len(portfolio_df)),
        'rejected': 0,
        'reasons': {},
        'note': 'E6 skeleton – no caps or brakes applied'
    }
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)

    print(f"Portfolio run complete. Bets: {len(portfolio_df)}. Outputs written to {bets_path}, {metrics_path}, {audit_path}.")


if __name__ == '__main__':
    main()
