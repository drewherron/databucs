"""batter_profile: how each batter handles different pitches, as of each game day.

One row per batter per day after a day they batted, keyed on (batter_id, as_of_date).
as_of_date is the first day the row may be used: the day after the last game day it
includes. A point-in-time lookup for a game on day D takes the newest row with
as_of_date <= D, so it only ever sees pitches from before D.

The profile covers the trailing WINDOW_DAYS of pitches and splits them into cells:

  pitch group  fastball / breaking / offspeed (from fact_pitch.pitch_group)
  region       zone (Statcast zones 1-9) / chase (outside the zone)
  count group  hitter_ahead (more balls than strikes, fewer than two strikes)
               two_strikes
               neutral (0-0, 0-1, 1-1)

Per cell: swing rate (in the chase region, that is the chase rate), whiff rate per swing,
and xwOBA on contact (Statcast's expected wOBA on balls in play, a damage measure that
ignores defense and luck). Each rate is shrunk toward the league rate for the same cell
over the same window, so a batter with few pitches in a cell looks close to average. The
prior sizes below are judgment calls, in the units of each rate's denominator.
"""

from __future__ import annotations

from itertools import product

from databucs.features.common import Source, date_filter, shrunk, trailing_window

TABLE = "batter_profile"
PRIMARY_KEYS = ["batter_id", "as_of_date"]
TIMESERIES_COLUMN = "as_of_date"

PITCH_GROUPS = ("fastball", "breaking", "offspeed")
REGIONS = {"zone": "is_in_zone", "chase": "NOT is_in_zone"}
COUNT_GROUPS = {
    "hitter_ahead": "balls > strikes AND strikes < 2",
    "two_strikes": "strikes = 2",
    "neutral": "balls <= strikes AND strikes < 2",
}

# Prior sizes for shrinkage, in pitches, swings, balls in play and plate appearances
SWING_PRIOR = 40
WHIFF_PRIOR = 30
CONTACT_PRIOR = 25
PA_PRIOR = 60

# Bat speed and swing length average only swings at 50 mph or more, which leaves out
# bunts and most checked swings
FULL_SWING_MIN_BAT_SPEED = 50


def cells() -> list[tuple[str, str]]:
    """(name, SQL condition) for every pitch group x region x count group cell."""
    return [
        (
            f"{group}_{region}_{count}",
            f"pitch_group = '{group}' AND {REGIONS[region]} AND {COUNT_GROUPS[count]}",
        )
        for group, region, count in product(PITCH_GROUPS, REGIONS, COUNT_GROUPS)
    ]


def _counters() -> list[tuple[str, str]]:
    """(name, SQL expression) for every per-pitch counter summed by day and window."""
    counters = [
        ("pitches", "1"),
        ("zone_pitches", "if(is_in_zone, 1, 0)"),
        ("zone_swings", "if(is_in_zone AND is_swing, 1, 0)"),
        ("chase_pitches", "if(NOT is_in_zone, 1, 0)"),
        ("chase_swings", "if(NOT is_in_zone AND is_swing, 1, 0)"),
        ("swings", "if(is_swing, 1, 0)"),
        ("whiffs", "if(is_whiff, 1, 0)"),
        ("contact_n", "if(is_in_play AND expected_woba IS NOT NULL, 1, 0)"),
        ("contact_xwoba", "if(is_in_play, coalesce(expected_woba, 0), 0)"),
        ("pa", "if(is_final_pitch_of_pa, 1, 0)"),
        ("strikeouts", "if(events IN ('strikeout', 'strikeout_double_play'), 1, 0)"),
        ("walks", "if(events IN ('walk', 'intent_walk'), 1, 0)"),
        ("full_swings", f"if(bat_speed >= {FULL_SWING_MIN_BAT_SPEED}, 1, 0)"),
        ("bat_speed_sum", f"if(bat_speed >= {FULL_SWING_MIN_BAT_SPEED}, bat_speed, 0)"),
        ("swing_length_sum", f"if(bat_speed >= {FULL_SWING_MIN_BAT_SPEED}, swing_length, 0)"),
    ]
    for name, condition in cells():
        counters += [
            (f"{name}_pitches", f"if({condition}, 1, 0)"),
            (f"{name}_swings", f"if({condition} AND is_swing, 1, 0)"),
            (f"{name}_whiffs", f"if({condition} AND is_whiff, 1, 0)"),
            (
                f"{name}_contact_n",
                f"if({condition} AND is_in_play AND expected_woba IS NOT NULL, 1, 0)",
            ),
            (
                f"{name}_contact_xwoba",
                f"if({condition} AND is_in_play, coalesce(expected_woba, 0), 0)",
            ),
        ]
    return counters


def _features() -> list[str]:
    """SQL select items for the published features (b = batter window, lg = league)."""

    def rate(num: str, den: str, prior: int) -> str:
        return shrunk(f"b.{num}", f"b.{den}", f"lg.{num} / lg.{den}", prior)

    items = [
        "b.pitches AS pitches_seen",
        "b.pa AS plate_appearances",
        f"{rate('zone_swings', 'zone_pitches', SWING_PRIOR)} AS zone_swing_rate",
        f"{rate('chase_swings', 'chase_pitches', SWING_PRIOR)} AS chase_rate",
        f"{rate('whiffs', 'swings', WHIFF_PRIOR)} AS whiff_rate",
        f"{rate('contact_xwoba', 'contact_n', CONTACT_PRIOR)} AS xwoba_contact",
        f"{rate('strikeouts', 'pa', PA_PRIOR)} AS strikeout_rate",
        f"{rate('walks', 'pa', PA_PRIOR)} AS walk_rate",
        "b.full_swings AS full_swings_tracked",
        "try_divide(b.bat_speed_sum, b.full_swings) AS bat_speed",
        "try_divide(b.swing_length_sum, b.full_swings) AS swing_length",
    ]
    for name, _ in cells():
        items += [
            f"b.{name}_pitches AS {name}_pitches",
            f"{rate(f'{name}_swings', f'{name}_pitches', SWING_PRIOR)} AS {name}_swing_rate",
            f"{rate(f'{name}_whiffs', f'{name}_swings', WHIFF_PRIOR)} AS {name}_whiff_rate",
            f"{rate(f'{name}_contact_xwoba', f'{name}_contact_n', CONTACT_PRIOR)}"
            f" AS {name}_xwoba_contact",
        ]
    return items


def build_sql(source: Source) -> str:
    """SQL for the batter_profile rows for game days in [source.start, source.end]."""
    counters = _counters()
    day_sums = ",\n    ".join(f"sum({expr}) AS {name}" for name, expr in counters)
    league_day_sums = ",\n    ".join(f"sum({name}) AS {name}" for name, _ in counters)
    batter_windows = ",\n    ".join(
        f"sum({name}) OVER {trailing_window('batter_id')} AS {name}" for name, _ in counters
    )
    league_windows = ",\n    ".join(
        f"sum({name}) OVER {trailing_window(None)} AS {name}" for name, _ in counters
    )
    features = ",\n  ".join(_features())
    return f"""
WITH pitches AS (
  SELECT *
  FROM {source.fact_pitch}
  WHERE NOT is_pitch_clock_violation
    AND {date_filter("game_date", source.history_start(), source.end)}
),
batter_days AS (
  SELECT
    batter_id,
    game_date,
    {day_sums}
  FROM pitches
  GROUP BY batter_id, game_date
),
league_days AS (
  SELECT
    game_date,
    {league_day_sums}
  FROM batter_days
  GROUP BY game_date
),
batter_windows AS (
  SELECT
    batter_id,
    game_date,
    {batter_windows}
  FROM batter_days
),
league_windows AS (
  SELECT
    game_date,
    {league_windows}
  FROM league_days
)
SELECT
  b.batter_id,
  date_add(b.game_date, 1) AS as_of_date,
  b.game_date AS batter_data_through,
  pl.bats AS batter_bats,
  {features}
FROM batter_windows b
JOIN league_windows lg USING (game_date)
LEFT JOIN {source.players} pl ON pl.player_id = b.batter_id
WHERE {date_filter("b.game_date", source.start, source.end)}
"""
