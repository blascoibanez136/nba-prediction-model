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

    This implementation enforces the full E6 risk controls defined in the policy:

    * **Per-bet cap** and **daily stake cap** as in Step 5.
    * **Team exposure cap** per day.
    * **Kill-switch/halt logic** (as before).
    * **Drawdown brakes**: Adjust the effective Kelly fraction and per-bet cap
      when cumulative drawdown exceeds specified thresholds. Later stages may
      halt betting for a cooldown period.
    * **Loss-streak dampening**: Scale down stakes multiplicatively when
      consecutive losses accrue, following a defined schedule.

    The function returns a copy of the input DataFrame with two new columns:
    ``stake_u_final`` (the final stake) and ``e6_reason`` (reason code).
    """
    # Create a working copy
    df = df.copy()

    # Extract policy components
    risk_caps = policy.get('risk_caps', {})
    max_daily_default = float(risk_caps.get('max_daily_stake_u', float('inf')))
    max_team_default = float(risk_caps.get('max_team_exposure_u', float('inf')))
    max_per_bet_default = float(risk_caps.get('max_per_bet_stake_u', float('inf')))

    stake_model = policy.get('stake_model', {})
    base_kelly = float(stake_model.get('kelly_fraction', 1.0))

    drawdown_cfg = policy.get('drawdown_brakes', {}) if isinstance(policy, dict) else {}
    drawdown_enabled = drawdown_cfg.get('enabled', False)
    drawdown_stages = drawdown_cfg.get('stages', []) if drawdown_enabled else []
    # Sort stages by increasing drawdown threshold
    drawdown_stages = sorted(drawdown_stages, key=lambda s: s.get('drawdown_u', 0.0))

    loss_cfg = policy.get('loss_streak_dampening', {}) if isinstance(policy, dict) else {}
    loss_enabled = loss_cfg.get('enabled', False)
    loss_schedule = loss_cfg.get('schedule', []) if loss_enabled else []
    # Build a lookup of losses -> multiplier; fill missing values later
    loss_multiplier_map = {}
    if loss_enabled:
        for entry in loss_schedule:
            loss_multiplier_map[int(entry['losses'])] = float(entry['multiplier'])
        floor_mult = float(loss_cfg.get('floor_multiplier', 1.0))
    else:
        floor_mult = 1.0

    # Kill-switch configuration
    kill_switch_cfg = policy.get('kill_switch', {}) if isinstance(policy, dict) else {}
    kill_switch_enabled = kill_switch_cfg.get('enabled', False)
    kill_dates: set[str] = set()
    if kill_switch_enabled:
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
                for entry in diag:
                    date = entry.get('date')
                    reasons = entry.get('reasons', [])
                    if date and any(r in supported_reasons for r in reasons):
                        kill_dates.add(date)
                        if cooldown:
                            try:
                                ts = pd.to_datetime(date)
                                for i in range(1, cooldown + 1):
                                    cd = (ts + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                                    kill_dates.add(cd)
                            except Exception:
                                pass
            except Exception:
                pass

    # Tracking structures
    daily_stake: dict[str, float] = {}
    team_exposure: dict[tuple[str, str], float] = {}

    # Dynamic state for drawdown and halting
    current_kelly = base_kelly
    current_max_per_bet = max_per_bet_default
    halt_until: str | None = None
    cum_profit = 0.0
    peak_profit = 0.0
    loss_streak = 0

    # Sort by date ascending and EV descending to allocate caps to highest EV first
    if 'ev_units' in df.columns:
        sorted_df = df.sort_values(by=['game_date', 'ev_units'], ascending=[True, False])
    else:
        sorted_df = df.sort_values(by=['game_date'])

    out_rows = []
    for _, row in sorted_df.iterrows():
        row_out = row.copy()
        date = str(row.get('game_date') or row.get('date'))
        reason = 'accepted'

        # Determine if kill-switch or drawdown halt applies
        # Convert date string to comparable object for halt range comparison
        if halt_until is not None and date <= halt_until:
            # Portfolio halt due to drawdown stage
            row_out['stake_u_final'] = 0.0
            row_out['e6_reason'] = 'halt_drawdown'
            out_rows.append(row_out)
            continue
        if kill_switch_enabled and date in kill_dates:
            row_out['stake_u_final'] = 0.0
            row_out['e6_reason'] = 'kill_switch_halt'
            out_rows.append(row_out)
            continue

        # Base stake from stake_u (default 1.0 if missing)
        base_stake = float(row.get('stake_u', 1.0))
        # Apply current per-bet cap
        base_stake = min(base_stake, current_max_per_bet)

        # Determine drawdown-based Kelly multiplier
        if base_kelly > 0:
            kelly_multiplier = current_kelly / base_kelly
        else:
            kelly_multiplier = 1.0

        # Determine loss-streak multiplier
        if loss_enabled:
            mult = loss_multiplier_map.get(loss_streak)
            if mult is None:
                # If schedule doesn't specify this loss count, use floor
                mult = floor_mult
            loss_multiplier = mult
        else:
            loss_multiplier = 1.0

        # Compute tentative stake after dynamic multipliers
        stake_dynamic = base_stake * kelly_multiplier * loss_multiplier

        # Enforce daily cap
        used_daily = daily_stake.get(date, 0.0)
        if used_daily + stake_dynamic > max_daily_default:
            stake_final = 0.0
            reason = 'daily_cap'
        else:
            # Enforce team exposure caps
            home_team = row.get('home_team')
            away_team = row.get('away_team')
            teams = []
            if pd.notnull(home_team):
                teams.append(str(home_team))
            if pd.notnull(away_team):
                teams.append(str(away_team))
            exceed = False
            if teams:
                for team in teams:
                    key = (date, team)
                    used = team_exposure.get(key, 0.0)
                    if used + stake_dynamic > max_team_default:
                        exceed = True
                        break
            if exceed:
                stake_final = 0.0
                reason = 'team_cap'
            else:
                stake_final = stake_dynamic
                reason = 'accepted'
                # Update exposures
                daily_stake[date] = used_daily + stake_final
                if teams:
                    for team in teams:
                        key = (date, team)
                        team_exposure[key] = team_exposure.get(key, 0.0) + stake_final

        # Record final stake and reason
        row_out['stake_u_final'] = stake_final
        row_out['e6_reason'] = reason
        out_rows.append(row_out)

        # Update running performance metrics only if stake_final > 0 and result/profit exist
        if stake_final > 0:
            # Determine profit per unit (row['profit'] column is per 1u stake)
            profit_per_u = None
            if 'profit' in row:
                try:
                    profit_per_u = float(row['profit'])
                except Exception:
                    profit_per_u = None
            if profit_per_u is None and 'result' in row and 'price' in row:
                try:
                    price = float(row['price'])
                    result_label = row['result']
                    if isinstance(result_label, str):
                        # Map result strings to 1/-1/0
                        if result_label.lower() == 'win':
                            outcome = 1
                        elif result_label.lower() == 'loss':
                            outcome = -1
                        else:
                            outcome = 0
                    else:
                        outcome = float(result_label)
                    # Profit per unit for given price: winning yields 100/|price|, losing yields -1, push yields 0
                    if outcome > 0:
                        profit_per_u = (100.0 / abs(price))
                    elif outcome < 0:
                        profit_per_u = -1.0
                    else:
                        profit_per_u = 0.0
                except Exception:
                    profit_per_u = None
            if profit_per_u is not None:
                profit_adj = profit_per_u * stake_final
                cum_profit += profit_adj
                # Update loss streak: negative profit counts as a loss
                if profit_adj < 0:
                    loss_streak += 1
                elif profit_adj > 0:
                    loss_streak = 0
                # Update peak and drawdown
                if cum_profit > peak_profit:
                    peak_profit = cum_profit
                drawdown = peak_profit - cum_profit
                # Evaluate drawdown brakes (only after this bet)
                if drawdown_enabled:
                    for stage in drawdown_stages:
                        threshold = float(stage.get('drawdown_u', 0.0))
                        if drawdown >= threshold:
                            action = stage.get('action')
                            if action == 'reduce_kelly':
                                # Reduce Kelly fraction if new_kelly is lower than current
                                new_kelly = float(stage.get('new_kelly', current_kelly))
                                if new_kelly < current_kelly:
                                    current_kelly = new_kelly
                            elif action == 'cap_max_stake':
                                new_max = float(stage.get('new_max_per_bet_stake_u', current_max_per_bet))
                                if new_max < current_max_per_bet:
                                    current_max_per_bet = new_max
                            elif action == 'halt':
                                # Set halt until date + cooldown days
                                cooldown_days = int(stage.get('cooldown_days', 0))
                                # Determine next date string
                                try:
                                    ts = pd.to_datetime(date)
                                    halt_until = (ts + pd.Timedelta(days=cooldown_days)).strftime('%Y-%m-%d')
                                except Exception:
                                    halt_until = date
                        # Continue evaluating next stages (to allow multiple actions at different thresholds)

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
