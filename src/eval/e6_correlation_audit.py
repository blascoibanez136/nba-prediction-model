import argparse
import json
import pandas as pd
from pathlib import Path


def load_candidates(candidate_path: Path) -> pd.DataFrame:
    """Load candidate bets from CSV and harmonize column names."""
    df = pd.read_csv(candidate_path)
    # Harmonize date column naming
    if 'game_date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'game_date'})
    return df


def compute_daily_metrics(df: pd.DataFrame):
    """Compute daily bet counts, daily team exposures, top team exposures, and daily PnL."""
    # Daily bet counts
    daily_counts = df.groupby('game_date').size()

    # Team exposures: count appearances of each team across home and away columns
    # Melt the DataFrame so both home_team and away_team contribute to exposure counts
    if 'home_team' in df.columns and 'away_team' in df.columns:
        exposures = df.melt(id_vars=['game_date'], value_vars=['home_team', 'away_team'],
                            var_name='role', value_name='team')
        daily_team_exposure = exposures.groupby(['game_date', 'team']).size().rename('exposure')
        top_team_exposure = exposures.groupby('team').size().rename('total_exposure')
    else:
        daily_team_exposure = pd.Series(dtype='int')
        top_team_exposure = pd.Series(dtype='int')

    # Daily profit (if profit column exists)
    if 'profit' in df.columns:
        daily_pnl = df.groupby('game_date')['profit'].sum()
    else:
        daily_pnl = pd.Series(dtype='float')

    return daily_counts, daily_team_exposure, top_team_exposure, daily_pnl


def write_outputs(daily_counts, daily_team_exposure, top_team_exposure, daily_pnl, report_path: Path, concentration_path: Path):
    """Write summary report JSON and concentration CSV."""
    # Compute summary statistics
    report = {
        'day_count': int(len(daily_counts)),
        'mean_bets_per_day': float(daily_counts.mean()) if not daily_counts.empty else 0.0,
        'max_bets_in_a_day': int(daily_counts.max()) if not daily_counts.empty else 0,
        'min_bets_in_a_day': int(daily_counts.min()) if not daily_counts.empty else 0,
    }

    # Compute daily PnL autocorrelation (lag 1) if possible
    if len(daily_pnl) > 1:
        report['daily_pnl_autocorrelation_lag1'] = float(daily_pnl.autocorr(lag=1))
    else:
        report['daily_pnl_autocorrelation_lag1'] = None

    # Write report JSON
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Prepare concentration CSV: merge daily team exposure into a DataFrame
    if not daily_team_exposure.empty:
        concentration_df = daily_team_exposure.reset_index()
        concentration_df.columns = ['game_date', 'team', 'exposure']
    else:
        concentration_df = pd.DataFrame(columns=['game_date', 'team', 'exposure'])

    # Append overall team exposure ranking to the end of the CSV as separate entries
    if not top_team_exposure.empty:
        top_df = top_team_exposure.sort_values(ascending=False).reset_index()
        top_df.columns = ['team', 'total_exposure']
        # Add an indicator column for overall exposure and join with date as NA
        top_df['game_date'] = 'ALL'
        top_df['exposure'] = top_df['total_exposure']
        concentration_df = pd.concat([concentration_df, top_df[['game_date', 'team', 'exposure']]], ignore_index=True)

    concentration_path.parent.mkdir(parents=True, exist_ok=True)
    concentration_df.to_csv(concentration_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='E6 Correlation & Concentration Audit')
    parser.add_argument('--candidates', default='outputs/ats_e3_staked_bets.csv', help='Path to candidate bets CSV')
    parser.add_argument('--report', default='outputs/e6_correlation_report.json', help='Path to write JSON summary report')
    parser.add_argument('--concentration_csv', default='outputs/e6_concentration_report.csv', help='Path to write exposure CSV')
    args = parser.parse_args()

    candidate_path = Path(args.candidates)
    df = load_candidates(candidate_path)
    daily_counts, daily_team_exposure, top_team_exposure, daily_pnl = compute_daily_metrics(df)
    write_outputs(daily_counts, daily_team_exposure, top_team_exposure, daily_pnl,
                  Path(args.report), Path(args.concentration_csv))
    print(f"Wrote correlation report to {args.report}\nWrote concentration report to {args.concentration_csv}")


if __name__ == '__main__':
    main()
