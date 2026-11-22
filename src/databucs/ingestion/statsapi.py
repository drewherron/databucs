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


# Postponed (D) and cancelled (C) games are also marked abstractGameState "Final".
NOT_PLAYED_CODES = {"D", "C"}


def was_played(game: dict) -> bool:
    status = game.get("status", {})
    return (
        status.get("abstractGameState") == "Final"
        and status.get("codedGameState") not in NOT_PLAYED_CODES
    )


def completed_game_dates(schedule_json: bytes) -> list[dt.date]:
    """Official dates of the games in a schedule response that were actually played.

    A resumed game is listed on the day it was finished, but its official date (the one
    Statcast files its pitches under) is the day it started.
    """
    payload = json.loads(schedule_json)
    dates = {
        dt.date.fromisoformat(game.get("officialDate", day["date"]))
        for day in payload.get("dates", [])
        for game in day["games"]
        if was_played(game)
    }
    return sorted(dates)
