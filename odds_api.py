import os
import requests
from pathlib import Path

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

def main():
    if not ODDS_API_KEY:
        raise SystemExit("ODDS_API_KEY not set")
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "odds_latest.json").write_text(json.dumps(data))
    print(f"saved {len(data)} games of odds")

if __name__ == "__main__":
    import json
    main()
