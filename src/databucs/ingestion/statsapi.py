"""MLB Stats API (statsapi.mlb.com): schedule, teams, players."""

from __future__ import annotations

import datetime as dt
import json
import urllib.parse

BASE_URL = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1

# Same game types as the Statcast pull: regular season + postseason.
GAME_TYPES = ("R", "F", "D", "L", "W")


def schedule_url(start: dt.date, end: dt.date) -> str:
    params = {
        "sportId": MLB_SPORT_ID,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "gameType": ",".join(GAME_TYPES),
        "hydrate": "probablePitcher,lineups",
    }
    return f"{BASE_URL}/schedule?{urllib.parse.urlencode(params)}"


def teams_url(season: int) -> str:
    return f"{BASE_URL}/teams?{urllib.parse.urlencode({'sportId': MLB_SPORT_ID, 'season': season})}"


def players_url(season: int) -> str:
    return f"{BASE_URL}/sports/{MLB_SPORT_ID}/players?{urllib.parse.urlencode({'season': season})}"


def completed_game_dates(schedule_json: bytes) -> list[dt.date]:
    """Dates in a schedule response with at least one completed (Final) game."""
    payload = json.loads(schedule_json)
    dates = []
    for day in payload.get("dates", []):
        if any(g.get("status", {}).get("abstractGameState") == "Final" for g in day["games"]):
            dates.append(dt.date.fromisoformat(day["date"]))
    return sorted(dates)
