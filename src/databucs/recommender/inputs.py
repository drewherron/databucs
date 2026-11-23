"""Who and what the recommender scores: the target series, the opposing lineup, the Pirates
pitchers, and one row per hypothetical pitch.

Every query reads only data from before the series starts (game_date < series_start), and
the feature lookups use the series start as the game date, so a demo on a past series sees
what a real run that morning would have seen.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from databucs.features import batter_profile
from databucs.models import features
from databucs.models.outcomes import COMPETITIVE

PIRATES_TEAM_ID = 134

# Arsenal: pitch types a pitcher actually uses (trailing 365 days, from the arsenal table)
MIN_USAGE_RATE = 0.03
MIN_PITCHES_THROWN = 25
# Projected lineup: batters who started at least MIN_STARTS of the opponent's last
# RECENT_GAMES games
RECENT_GAMES = 10
MIN_STARTS = 3
# Pitchers: everyone who pitched in the team's last RECENT_TEAM_GAMES games, plus the
# series' probable starters
RECENT_TEAM_GAMES = 15
HISTORY_DAYS = 365  # window for strike zones, handedness and pitch usage
# Usual mix: by exact count when the pitcher has this many pitches there vs the side
MIN_COUNT_PITCHES = 40

# The hypothetical pitch's own columns (features.PITCH_COLUMNS), from the pitcher's
# arsenal row (a), the matchup (m) and the zone point (z). The arsenal's horizontal values
# are already arm side positive, like the model's.
HYPOTHETICAL_PITCH = {
    "pitch_type": "a.pitch_type",
    "pitch_group": "g.pitch_group",
    "pitcher_throws": "m.pitcher_throws",
    "batter_stands": "m.batter_stands",
    "is_in_zone": "z.zone <= 9",
    "plate_x": "z.plate_x",
    "plate_z": "m.sz_bot + z.height * (m.sz_top - m.sz_bot)",
    "sz_top": "m.sz_top",
    "sz_bot": "m.sz_bot",
    "pitch_velocity": "a.velocity",
    "pitch_spin_rate": "a.spin_rate",
    "pitch_horizontal_break": "a.horizontal_break",
    "pitch_induced_vertical_break": "a.induced_vertical_break",
    "pitch_release_side": "a.release_side",
    "pitch_release_height": "a.release_height",
    "pitch_extension": "a.extension",
    "pitch_arm_angle": "a.arm_angle",
}

OPTION_KEY = ["pitcher_id", "batter_id", "scenario", "pitch_type", "zone"]
POINT_KEY = [*OPTION_KEY, "point"]


@dataclass(frozen=True)
class Game:
    game_pk: int
    official_date: dt.date
    home_team_id: int
    away_team_id: int
    is_played: bool
    series_game_number: int | None
    home_probable_pitcher_id: int | None = None
    away_probable_pitcher_id: int | None = None

    def opponent_of(self, team_id: int) -> int:
        return self.away_team_id if self.home_team_id == team_id else self.home_team_id

    def probable_for(self, team_id: int) -> int | None:
        if self.home_team_id == team_id:
            return self.home_probable_pitcher_id
        return self.away_probable_pitcher_id


@dataclass(frozen=True)
class Target:
    team_id: int
    opponent_id: int
    series_start: dt.date
    mode: str  # scheduled, demo, chosen or custom (no scheduled game)
    game_pks: list[int] = field(default_factory=list)
    probable_starters: list[int] = field(default_factory=list)


def choose_series(
    games: list[Game],
    team_id: int,
    today: dt.date,
    max_days_ahead: int,
    series_start: dt.date | None = None,
    opponent_id: int | None = None,
) -> Target:
    """Pick the series to prepare for.

    - series_start given: the team's game that day (against opponent_id, if given). With
      no such game, a custom series against opponent_id, e.g. a future game not on the
      schedule yet.
    - otherwise the next unplayed series starting within max_days_ahead of today
      ("scheduled"), or, failing that, the last series already played ("demo").
    """
    team_games = sorted(
        (g for g in games if team_id in (g.home_team_id, g.away_team_id)),
        key=lambda g: (g.official_date, g.game_pk),
    )
    if series_start is not None:
        first = next(
            (
                g
                for g in team_games
                if g.official_date == series_start and opponent_id in (None, g.opponent_of(team_id))
            ),
            None,
        )
        if first is None:
            if opponent_id is None:
                raise ValueError(f"No game on {series_start}; pass an opponent for a custom series")
            return Target(team_id, opponent_id, series_start, "custom")
        mode = "chosen"
    else:
        horizon = today + dt.timedelta(days=max_days_ahead)
        upcoming = [
            g for g in team_games if not g.is_played and today <= g.official_date <= horizon
        ]
        if upcoming:
            first, mode = upcoming[0], "scheduled"
        else:
            openers = [g for g in team_games if g.is_played and g.series_game_number == 1]
            if not openers:
                raise ValueError("No upcoming or past series found")
            first, mode = openers[-1], "demo"

    # The series: consecutive games against the same opponent from the first one
    opponent = first.opponent_of(team_id)
    series = []
    for g in team_games[team_games.index(first) :]:
        if g.opponent_of(team_id) != opponent:
            break
        series.append(g)
    starters = [p for g in series if (p := g.probable_for(team_id)) is not None]
    return Target(
        team_id,
        opponent,
        first.official_date,
        mode,
        [g.game_pk for g in series],
        list(dict.fromkeys(starters)),
    )


def games_sql(games_table: str, team_id: int) -> str:
    return f"""
SELECT game_pk, official_date, home_team_id, away_team_id, is_played, series_game_number,
  home_probable_pitcher_id, away_probable_pitcher_id
FROM {games_table}
WHERE {team_id} IN (home_team_id, away_team_id) AND game_type IN ('R', 'F', 'D', 'L', 'W')
"""


def projected_lineup_sql(games_table: str, lineups_table: str, target: Target) -> str:
    """The opponent's regulars: batters with MIN_STARTS of its last RECENT_GAMES starts,
    with their most common batting order spot."""
    return f"""
WITH recent AS (
  SELECT game_pk
  FROM {games_table}
  WHERE {target.opponent_id} IN (home_team_id, away_team_id)
    AND is_played AND official_date < DATE'{target.series_start}'
  ORDER BY official_date DESC, game_number DESC
  LIMIT {RECENT_GAMES}
)
SELECT player_id AS batter_id, mode(batting_order) AS lineup_spot, count(*) AS recent_starts
FROM {lineups_table}
JOIN recent USING (game_pk)
WHERE team_id = {target.opponent_id}
GROUP BY player_id
HAVING count(*) >= {MIN_STARTS}
"""


def recent_pitchers_sql(
    fact_pitch: str, games_table: str, players_table: str, target: Target
) -> str:
    """Everyone who pitched for the team in its last RECENT_TEAM_GAMES games (position
    players who mopped up a blowout excluded)."""
    return f"""
WITH recent AS (
  SELECT game_pk
  FROM {games_table}
  WHERE {target.team_id} IN (home_team_id, away_team_id)
    AND is_played AND official_date < DATE'{target.series_start}'
  ORDER BY official_date DESC, game_number DESC
  LIMIT {RECENT_TEAM_GAMES}
)
SELECT DISTINCT f.pitcher_id
FROM {fact_pitch} f
JOIN recent USING (game_pk)
JOIN {players_table} pl ON pl.player_id = f.pitcher_id
WHERE f.pitching_team_id = {target.team_id}
  AND pl.position_type IN ('Pitcher', 'Two-Way Player')
"""


def _history(target: Target) -> str:
    start = f"DATE'{target.series_start}'"
    return f"game_date >= date_sub({start}, {HISTORY_DAYS}) AND game_date < {start}"


def request_rows_sql(
    fact_pitch: str, players_table: str, arsenal_table: str, target: Target
) -> str:
    """One row per (pitcher, batter, scenario, pitch type, zone, point): the models' inputs
    plus labels. Reads the temp views target_lineup, target_pitchers, scenarios and
    zone_points."""
    pitch = ",\n  ".join(f"{expr} AS {name}" for name, expr in HYPOTHETICAL_PITCH.items())
    state = ",\n  ".join(f"s.{name}" for name in features.STATE_COLUMNS)
    history = _history(target)
    return f"""
WITH batters AS (
  SELECT l.batter_id, l.lineup_spot, pl.full_name AS batter_name, pl.bats,
    coalesce(sz.sz_top, pl.strike_zone_top) AS sz_top,
    coalesce(sz.sz_bot, pl.strike_zone_bottom) AS sz_bot
  FROM target_lineup l
  LEFT JOIN {players_table} pl ON pl.player_id = l.batter_id
  LEFT JOIN (
    SELECT batter_id, median(sz_top) AS sz_top, median(sz_bot) AS sz_bot
    FROM {fact_pitch}
    WHERE {history}
    GROUP BY batter_id
  ) sz USING (batter_id)
),
-- The side each batter hits from against each pitcher hand (switch hitters change sides)
sides AS (
  SELECT batter_id, pitcher_throws, mode(batter_stands) AS batter_stands
  FROM {fact_pitch}
  WHERE {history}
  GROUP BY ALL
),
pitchers AS (
  SELECT t.pitcher_id, t.is_probable_starter, pl.full_name AS pitcher_name,
    coalesce(h.pitcher_throws, pl.throws) AS pitcher_throws
  FROM target_pitchers t
  LEFT JOIN {players_table} pl ON pl.player_id = t.pitcher_id
  LEFT JOIN (
    SELECT pitcher_id, mode(pitcher_throws) AS pitcher_throws
    FROM {fact_pitch}
    WHERE {history}
    GROUP BY ALL
  ) h USING (pitcher_id)
),
matchups AS (
  SELECT p.*, b.batter_id, b.lineup_spot, b.batter_name, b.sz_top, b.sz_bot,
    coalesce(
      s.batter_stands,
      CASE WHEN b.bats = 'S' THEN if(p.pitcher_throws = 'R', 'L', 'R') ELSE b.bats END
    ) AS batter_stands
  FROM pitchers p
  CROSS JOIN batters b
  LEFT JOIN sides s ON s.batter_id = b.batter_id AND s.pitcher_throws = p.pitcher_throws
),
-- Each pitcher's latest arsenal, limited to pitches they actually use
arsenal AS (
  SELECT a.*
  FROM {arsenal_table} a
  JOIN (
    SELECT pitcher_id, max(as_of_date) AS as_of_date
    FROM {arsenal_table}
    WHERE as_of_date <= DATE'{target.series_start}'
    GROUP BY pitcher_id
  ) latest USING (pitcher_id, as_of_date)
  WHERE a.usage_rate >= {MIN_USAGE_RATE} AND a.pitches_thrown >= {MIN_PITCHES_THROWN}
),
groups AS (
  SELECT pitch_type, mode(pitch_group) AS pitch_group
  FROM {fact_pitch}
  WHERE pitch_type IS NOT NULL
  GROUP BY pitch_type
)
SELECT
  m.pitcher_id, m.pitcher_name, m.is_probable_starter,
  m.batter_id, m.batter_name, m.lineup_spot,
  s.scenario, z.zone, z.point,
  a.pitches_thrown AS pitcher_pitches_thrown, a.usage_rate AS pitch_usage_rate,
  DATE'{target.series_start}' AS game_date,
  {pitch},
  {state}
FROM matchups m
JOIN arsenal a USING (pitcher_id)
JOIN groups g ON g.pitch_type = a.pitch_type
CROSS JOIN scenarios s
CROSS JOIN zone_points z
"""


def count_group_sql(balls: str = "balls", strikes: str = "strikes") -> str:
    """CASE expression giving the batter profile's count group for a count."""
    cases = " ".join(
        f"WHEN {cond.replace('balls', balls).replace('strikes', strikes)} THEN '{name}'"
        for name, cond in batter_profile.COUNT_GROUPS.items()
    )
    return f"CASE {cases} END"


def usage_sql(fact_pitch: str, target: Target) -> str:
    """Each target pitcher's mix of pitch type x zone for each scenario count, against
    same- or opposite-side hitters, over the trailing HISTORY_DAYS. This is "the pitcher's
    usual choice" the recommendations are compared with.

    It uses the exact count when the pitcher threw at least MIN_COUNT_PITCHES there against
    that side, and his count group otherwise. Pooling a whole group would be misleading:
    full count shares "two strikes" with 0-2, where pitchers throw far more chase pitches.
    Reads the temp views target_pitchers and scenarios.
    """
    return f"""
WITH pitches AS (
  SELECT pitcher_id, balls, strikes, {count_group_sql()} AS count_group,
    pitcher_throws = batter_stands AS same_side, pitch_type, zone
  FROM {fact_pitch}
  WHERE ({COMPETITIVE}) AND {_history(target)} AND zone IS NOT NULL
    AND pitcher_id IN (SELECT pitcher_id FROM target_pitchers)
),
count_totals AS (
  SELECT pitcher_id, balls, strikes, same_side, count(*) AS pitches
  FROM pitches
  GROUP BY ALL
),
situations AS (
  SELECT ps.pitcher_id, ps.same_side, c.balls, c.strikes,
    {count_group_sql("c.balls", "c.strikes")} AS count_group,
    coalesce(t.pitches, 0) >= {MIN_COUNT_PITCHES} AS by_exact_count
  FROM (SELECT DISTINCT pitcher_id, same_side FROM pitches) ps
  CROSS JOIN (SELECT DISTINCT balls, strikes FROM scenarios) c
  LEFT JOIN count_totals t USING (pitcher_id, same_side, balls, strikes)
)
SELECT s.pitcher_id, s.balls, s.strikes, s.same_side, p.pitch_type, p.zone,
  first(s.by_exact_count) AS usual_by_exact_count,
  count(*) / sum(count(*)) OVER (PARTITION BY s.pitcher_id, s.balls, s.strikes, s.same_side)
    AS usage
FROM situations s
JOIN pitches p
  ON p.pitcher_id = s.pitcher_id AND p.same_side = s.same_side
  AND CASE
    WHEN s.by_exact_count THEN p.balls = s.balls AND p.strikes = s.strikes
    ELSE p.count_group = s.count_group
  END
GROUP BY s.pitcher_id, s.balls, s.strikes, s.same_side, p.pitch_type, p.zone
"""
