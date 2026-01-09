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

    This implementation enforces basic risk controls defined in the policy:

    * **Per-bet cap**: Stake per bet is capped at `max_per_bet_stake_u`.
    * **Daily stake cap**: Total stake across all bets on a given date cannot
      exceed `max_daily_stake_u`. Bets beyond this cap are rejected.
    * **Team exposure cap**: Stake allocated to any single team (regardless
      of home/away side) on a given date cannot exceed `max_team_exposure_u`.
      If a bet would push exposure over the limit for either team, it is
      rejected.
    * **Kill-switch/halt logic**: If the policy includes a ``kill_switch``
      section and it is enabled, any game date present in the referenced
      kill-switch diagnostics with a supported reason (e.g. negative rolling
      CLV/ROI) will result in zero bets for that date. A cooldown period
      specified in the policy will also halt the following days.

    The function returns a copy of the input DataFrame with two new columns:

    * ``stake_u_final`` – the final stake applied after caps and halts.
    * ``e6_reason`` – a reason code indicating why the bet was accepted
      (``accepted``) or rejected (e.g. ``daily_cap``, ``team_cap``,
      ``kill_switch_halt``).

    Drawdown brakes, loss-streak dampening, and market exposure caps are
    placeholders for future steps and are not implemented here.
    """
    # Copy to avoid mutating caller data
    out_rows = []
    # Extract cap parameters with sensible defaults
    risk_caps = policy.get('risk_caps', {})
    max_per_bet = float(risk_caps.get('max_per_bet_stake_u', float('inf')))
    max_daily = float(risk_caps.get('max_daily_stake_u', float('inf')))
    max_team = float(risk_caps.get('max_team_exposure_u', float('inf')))

    # Prepare kill-switch flags
    kill_switch_cfg = policy.get('kill_switch', {}) if isinstance(policy, dict) else {}
    kill_switch_enabled = kill_switch_cfg.get('enabled', False)
    # Build set of dates to halt if kill-switch is enabled
    kill_dates: set[str] = set()
    if kill_switch_enabled:
        # Extract reasons that trigger the kill-switch
        supported_reasons = set(kill_switch_cfg.get('supported_reasons', []))
        source_artifacts = kill_switch_cfg.get('source_artifacts', [])
        cooldown = kill_switch_cfg.get('behavior', {}).get('cooldown_days', 0)
        for artifact in source_artifacts:
            try:
                path = Path(artifact)
                if not path.exists():
                    continue
                with path.open('r', encoding='utf-8') as f:
                    diag = json.load(f)
                # Expect list of dicts with 'date' and 'reasons'
                for entry in diag:
                    date = entry.get('date')
                    reasons = entry.get('reasons', [])
                    # If any reason is in supported_reasons, flag this date
                    if date and any(r in supported_reasons for r in reasons):
                        kill_dates.add(date)
                        # Apply cooldown: include subsequent days as flagged
                        if cooldown:
                            try:
                                ts = pd.to_datetime(date)
                                for i in range(1, cooldown + 1):
                                    cd = (ts + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                                    kill_dates.add(cd)
                            except Exception:
                                pass
            except Exception:
                # Silently ignore errors reading kill-switch files
                pass

    # Track daily and team exposures
    daily_stake: dict[str, float] = {}
    team_exposure: dict[tuple[str, str], float] = {}

    # Sort bets by game_date then descending EV units (if present)
    # This ensures higher EV bets are allocated cap space first
    if 'ev_units' in df.columns:
        sorted_df = df.sort_values(by=['game_date', 'ev_units'], ascending=[True, False])
    else:
        sorted_df = df.sort_values(by=['game_date'])

    for _, row in sorted_df.iterrows():
        # Prepare output row copy
        row_out = row.copy()
        date = str(row.get('game_date') or row.get('date'))
        # Default reason
        reason = 'accepted'

        # Determine base stake from stake_u or default to 1.0
        stake = float(row.get('stake_u', 1.0))
        # Apply per-bet cap
        stake_final = min(stake, max_per_bet)

        # Enforce kill-switch: if date flagged, reject all bets for this date
        if kill_switch_enabled and date in kill_dates:
            stake_final = 0.0
            reason = 'kill_switch_halt'
        else:
            # Enforce daily cap
            used_daily = daily_stake.get(date, 0.0)
            if used_daily + stake_final > max_daily:
                stake_final = 0.0
                reason = 'daily_cap'
            else:
                # Enforce team exposure cap if team info available
                home_team = row.get('home_team')
                away_team = row.get('away_team')
                teams = []
                if pd.notnull(home_team):
                    teams.append(str(home_team))
                if pd.notnull(away_team):
                    teams.append(str(away_team))
                exceed_team_cap = False
                if teams:
                    for team in teams:
                        key = (date, team)
                        used_team = team_exposure.get(key, 0.0)
                        if used_team + stake_final > max_team:
                            exceed_team_cap = True
                            break
                if exceed_team_cap:
                    stake_final = 0.0
                    reason = 'team_cap'
                else:
                    # Accept bet: update exposure trackers
                    daily_stake[date] = used_daily + stake_final
                    if teams:
                        for team in teams:
                            key = (date, team)
                            team_exposure[key] = team_exposure.get(key, 0.0) + stake_final

        row_out['stake_u_final'] = stake_final
        row_out['e6_reason'] = reason
        out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)
    return out_df


def compute_metrics(portfolio_df: pd.DataFrame, policy: dict) -> dict:
    """
    Compute simple portfolio-level metrics from the final portfolio DataFrame.

    Metrics reported:

    * ``bet_count`` – number of bets with positive final stake (i.e. accepted bets).
    * ``total_stake_u`` – sum of final stake units across all bets.
    * ``rejected_count`` – number of bets rejected (stake_u_final == 0).
    * ``total_profit_u`` – sum of profit column if present; zero otherwise.

    In later phases, this function should also compute drawdown statistics,
    exposure summaries, and guard violation counts for QA purposes.
    """
    # Count accepted and rejected bets based on final stake
    stake_col = portfolio_df.get('stake_u_final')
    if stake_col is not None:
        accepted = portfolio_df[stake_col > 0]
        rejected = portfolio_df[stake_col <= 0]
    else:
        # If missing stake_u_final, treat all bets as accepted
        accepted = portfolio_df
        rejected = pd.DataFrame(columns=portfolio_df.columns)
    metrics = {
        'bet_count': int(len(accepted)),
        'total_stake_u': float(accepted['stake_u_final'].sum()) if not accepted.empty else 0.0,
        'rejected_count': int(len(rejected)),
    }
    # If profit column present, compute total profit
    if 'profit' in portfolio_df.columns:
        metrics['total_profit_u'] = float(portfolio_df['profit'].sum())
    else:
        metrics['total_profit_u'] = 0.0
    return metrics
