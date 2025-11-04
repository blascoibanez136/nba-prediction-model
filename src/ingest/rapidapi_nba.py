import os
import requests
from pathlib import Path

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
BASE_URL = "https://api-nba-v1.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com",
}

def fetch_games(season: int):
    resp = requests.get(f"{BASE_URL}/games", headers=HEADERS, params={"season": season}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in range(2019, 2025):
        data = fetch_games(season)
        (out_dir / f"rapidapi_games_{season}.json").write_text(json.dumps(data))
        print(f"saved season {season}")

if __name__ == "__main__":
    import json
    if not RAPIDAPI_KEY:
        raise SystemExit("RAPIDAPI_KEY not set")
    main()
