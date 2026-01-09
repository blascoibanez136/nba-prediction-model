import argparse
import json
import pandas as pd
from pathlib import Path


def load_policy(policy_path: str) -> dict:
    """
    Load a JSON policy file from disk.
    """
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    with open(policy_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_candidate_bets(policy: dict) -> pd.DataFrame:
    """
    Read the candidate bets DataFrame from the first available source defined in
    the policy's interfaces.candidate_input.source_preference. This does not
    perform any staking or ranking logic; it simply loads the CSV.
    """
    source_pref = policy.get('interfaces', {}).get('candidate_input', {}).get('source_preference', [])
    for src in source_pref:
        path = Path(src)
        if path.exists():
            df = pd.read_csv(path)
            # Harmonize date column naming
            if 'game_date' not in df.columns and 'date' in df.columns:
                df = df.rename(columns={'date': 'game_date'})
            return df
    raise FileNotFoundError(f"None of the candidate input files were found: {source_pref}")


def validate_candidates(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """
    Validate the candidate bets DataFrame against the required columns and
    side/market policies defined in the portfolio policy. Raises ValueError on
    invalid input. Returns the DataFrame unchanged on success.
    """
    # Check required columns (one-of semantics)
    required_any = policy.get('interfaces', {}).get('candidate_input', {}).get('required_columns_any_of', {})
    for key, cols in required_any.items():
        if not any(col in df.columns for col in cols):
            raise ValueError(f"Missing required column for '{key}'; expected one of {cols}")

    # Check market enforcement (ATS-only in v1)
    markets = set(df.get('market', ['spread']))
    if markets != {'spread'}:
        raise ValueError(f"E6 ATS-only enforcement failed; found markets: {markets}")

    # Check side policy (away-only in v1)
    side_policy = policy.get('scope', {}).get('side_policy', {}).get('ATS', 'away_only')
    if side_policy == 'away_only':
        sides = set(df.get('bet_side', ['away']))
        if sides != {'away'}:
            raise ValueError(f"E6 away-only enforcement failed; found bet sides: {sides}")

    # Check American odds contract if price column present
    if 'price' in df.columns:
        invalid_odds = df['price'].dropna().astype(float).abs() < 100
        if invalid_odds.any():
            raise ValueError("Invalid odds detected; all American odds must satisfy abs(odds) >= 100")
    return df


def write_outputs(df: pd.DataFrame, policy: dict, note: str = '') -> None:
    """
    Write the E6 portfolio outputs: bets CSV, metrics JSON, audit JSON. Since
    this is a skeleton, the bets CSV is a pass-through of the candidate bets.
    """
    outputs = policy.get('interfaces', {}).get('outputs', {})
    bets_path = Path(outputs.get('bets_csv', 'outputs/e6_portfolio_bets.csv'))
    metrics_path = Path(outputs.get('metrics_json', 'outputs/e6_portfolio_metrics.json'))
    audit_path = Path(outputs.get('audit_json', 'outputs/e6_portfolio_audit.json'))

    # Ensure output directory exists
    bets_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    # Write bets CSV
    df.to_csv(bets_path, index=False)

    # Write stub metrics
    metrics = {
        'bet_count': int(len(df)),
        'note': note or 'pass-through skeleton (no allocation changes)'
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Write stub audit
    audit = {
        'selected': int(len(df)),
        'rejected': 0,
        'reasons': {}
    }
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)


def run_portfolio(policy_path: str) -> None:
    policy = load_policy(policy_path)
    df = read_candidate_bets(policy)
    df = validate_candidates(df, policy)
    write_outputs(df, policy)


def main():
    parser = argparse.ArgumentParser(description='E6 Portfolio Runner (pass-through skeleton)')
    parser.add_argument('--policy', default='configs/e6_portfolio_policy_v1.json', help='Path to portfolio policy JSON')
    args = parser.parse_args()
    run_portfolio(args.policy)


if __name__ == '__main__':
    main()
