def merge_game_and_odds(games: list, odds: list):
    # placeholder merge
    rows = []
    for g in games:
        teams = g.get("teams", {})
        home = teams.get("home", {}).get("name")
        away = teams.get("visitors", {}).get("name")
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "game_id": g.get("id") or g.get("gameId"),
            }
        )
    return rows
