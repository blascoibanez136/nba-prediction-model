"""
E6 Portfolio overlay module.

This module defines functions to apply the E6 portfolio-level policy overlay to
pre-staked candidate bets. It is intentionally implemented as a pass-through
skeleton for the first iteration of E6. Later versions should incorporate
portfolio drawdown brakes, loss-streak dampening, exposure caps, and kill-switch
logic as defined in the policy file.

Functions:
- load_policy(path) -> dict: Load portfolio policy from JSON.
- apply_e6_portfolio(df, policy) -> pd.DataFrame: Apply E6 overlay (stub).
- compute_metrics(df, policy) -> dict: Compute simple portfolio metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def load_policy(policy_path: str | Path) -> dict:
    """Load a JSON policy file."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio policy not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def apply_e6_portfolio(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """
    Apply the E6 portfolio overlay to a pre-staked candidate bets DataFrame.

    **Skeleton implementation:**
    - Currently returns a shallow copy of `df` with a new column
      `stake_u_final` equal to existing `stake_u` (if present) or 0.
    - Sets a `reason` column to 'accepted' for all bets.

    Later versions should:
    - enforce per-bet caps (`max_per_bet_stake_u`), daily caps,
      team/market exposure caps, and bet-rate caps.
    - apply drawdown brakes and loss-streak dampening to modify effective
      Kelly fraction.
    - incorporate kill-switch/halt logic based on E5 diagnostics.
    """
    out = df.copy()
    # Determine initial stake; if `stake_u` not present, assume 1.0 per bet
    if 'stake_u' in out.columns:
        out['stake_u_final'] = out['stake_u']
    else:
        # If no stake_u provided, default to 1u (or 0 if policy disables)
        default_stake = 1.0
        out['stake_u_final'] = default_stake

    # Assign reason codes; initially all accepted
    out['e6_reason'] = 'accepted'

    return out


def compute_metrics(portfolio_df: pd.DataFrame, policy: dict) -> dict:
    """
    Compute simple portfolio-level metrics from the final portfolio DataFrame.

    Returns a dict with keys such as `bet_count`, `total_stake_u`, `total_profit_u`.
    Future versions should include drawdown metrics, guard violation counts, etc.
    """
    metrics = {
        'bet_count': int(len(portfolio_df)),
        'total_stake_u': float(portfolio_df.get('stake_u_final', 0.0).sum()),
    }
    # If profit column present, compute total profit
    if 'profit' in portfolio_df.columns:
        metrics['total_profit_u'] = float(portfolio_df['profit'].sum())
    return metrics
