"""pitcher_arsenal_features: what each of a pitcher's pitches looks like, as of each day.

One row per pitcher x pitch type per day after a day they pitched, keyed on
(pitcher_id, pitch_type, as_of_date), with the same point-in-time convention as
batter_profile: as_of_date is the day after the last game day included.

Averages cover the trailing WINDOW_DAYS of the pitcher's pitches. On each day a pitcher
pitched there is a row for every pitch type they threw in the window, not just that day,
so usage stays current for pitches they didn't throw that day.

Horizontal values are flipped for right-handers so positive always means arm side:
movement toward the pitcher's arm side and release toward the arm side of the rubber.
Movement is in inches (Statcast's pfx_x and pfx_z, in feet, times 12); vertical movement
is induced vertical break, the rise or drop compared with a spinless pitch.

The primary fastball is the most-thrown four-seamer, sinker or cutter in the window. Every
pitch type gets its speed and movement differences from it, which is how hitters
experience a secondary pitch.

Results over expected. The measures above describe a pitch; three more say how it has
worked, compared with what the league gets in the same spot, so that where and when a
pitcher throws it doesn't count as quality:
- whiff_over_expected: whiffs per swing minus the league's whiff rate per swing for the
  same pitch group, Statcast zone and count group
- chase_over_expected: swings per pitch outside the zone, minus the league's swing rate in
  the same cell (pitches in zones 11-14)
- xwoba_contact_over_expected: expected wOBA on balls in play, minus the league's for the
  same cell (negative is good for the pitcher)
Each is a sum of (actual - expected) over the window divided by the chances plus a prior,
which shrinks small samples toward 0 (league average). The league's cell rates are over the
same trailing window, so they only use data from before as_of_date.
"""

from __future__ import annotations

from databucs.features.common import Source, date_filter, trailing_window

TABLE = "pitcher_arsenal_features"
PRIMARY_KEYS = ["pitcher_id", "pitch_type", "as_of_date"]
TIMESERIES_COLUMN = "as_of_date"

ARM_SIDE = "if(pitcher_throws = 'R', -1, 1)"

# (feature name, per-pitch SQL expression) averaged over the window, ignoring NULLs
MEASURES = [
    ("velocity", "release_speed"),
    ("spin_rate", "release_spin_rate"),
    ("horizontal_break", f"pfx_x * 12 * {ARM_SIDE}"),
    ("induced_vertical_break", "pfx_z * 12"),
    ("release_side", f"release_pos_x * {ARM_SIDE}"),
    ("release_height", "release_pos_z"),
    ("extension", "release_extension"),
    ("arm_angle", "arm_angle"),
]
# Results over expected (see the module docstring): (feature, actual, expected,
# chances, prior), all per-pitch SQL over the `scored` rows below
RESULT_COUNT_GROUPS = {
    "hitter_ahead": "balls > strikes AND strikes < 2",
    "two_strikes": "strikes = 2",
    "neutral": "balls <= strikes AND strikes < 2",
}
RESULTS_OVER_EXPECTED = [
    ("whiff_over_expected", "if(is_swing AND is_whiff, 1, 0)",
     "if(is_swing, lg_whiff_rate, 0)", "if(is_swing, 1, 0)", 30),
    ("chase_over_expected", "if(NOT is_in_zone AND is_swing, 1, 0)",
     "if(NOT is_in_zone, lg_swing_rate, 0)", "if(NOT is_in_zone, 1, 0)", 40),
    ("xwoba_contact_over_expected", "if(has_contact_xwoba, expected_woba, 0)",
     "if(has_contact_xwoba, lg_xwoba_contact, 0)", "if(has_contact_xwoba, 1, 0)", 25),
]  # fmt: skip

FASTBALL_TYPES = ("FF", "SI", "FC")
FASTBALL_DIFFS = ("velocity", "horizontal_break", "induced_vertical_break")


def build_sql(source: Source) -> str:
    """SQL for the arsenal rows for game days in [source.start, source.end]."""
    day_sums = ",\n    ".join(
        [f"sum({expr}) AS {name}_sum,\n    count({expr}) AS {name}_n" for name, expr in MEASURES]
        + [
            f"sum({actual} - {expected}) AS {name}_sum,\n    sum({chances}) AS {name}_n"
            for name, actual, expected, chances, _ in RESULTS_OVER_EXPECTED
        ]
    )
    summed = [name for name, _ in MEASURES] + [name for name, *_ in RESULTS_OVER_EXPECTED]
    grid_values = ",\n    ".join(
        f"coalesce(d.{name}_sum, 0) AS {name}_sum,\n    coalesce(d.{name}_n, 0) AS {name}_n"
        for name in summed
    )
    window = trailing_window("g.pitcher_id, g.pitch_type")
    window_sums = ",\n    ".join(
        f"sum(g.{name}_sum) OVER {window} AS {name}_sum,\n"
        f"    sum(g.{name}_n) OVER {window} AS {name}_n"
        for name in summed
    )
    averages = ",\n    ".join(
        [f"try_divide({name}_sum, {name}_n) AS {name}" for name, _ in MEASURES]
        + [
            f"{name}_sum / ({name}_n + {prior}) AS {name}"
            for name, _, _, _, prior in RESULTS_OVER_EXPECTED
        ]
    )
    count_group = (
        "CASE "
        + " ".join(
            f"WHEN {condition} THEN '{group}'" for group, condition in RESULT_COUNT_GROUPS.items()
        )
        + " END"
    )
    league_window = trailing_window("pitch_group, zone, count_group")
    fastball_types = ", ".join(f"'{t}'" for t in FASTBALL_TYPES)
    diffs = ",\n  ".join(
        f"a.{name} - fb.{name} AS {name}_diff_from_fastball" for name in FASTBALL_DIFFS
    )
    measures = ",\n  ".join(
        [f"a.{name}" for name, _ in MEASURES] + [f"a.{name}" for name, *_ in RESULTS_OVER_EXPECTED]
    )
    return f"""
WITH all_pitches AS (
  SELECT *,
    {count_group} AS count_group,
    is_in_play AND expected_woba IS NOT NULL AS has_contact_xwoba
  FROM {source.fact_pitch}
  WHERE pitch_type IS NOT NULL
    AND {date_filter("game_date", source.history_start(), source.end)}
),
-- The league in each cell (pitch group x zone x count group), per day, then over the
-- trailing window ending that day
league_days AS (
  SELECT pitch_group, zone, count_group, game_date,
    count(*) AS pitches,
    sum(if(is_swing, 1, 0)) AS swings,
    sum(if(is_swing AND is_whiff, 1, 0)) AS whiffs,
    sum(if(has_contact_xwoba, 1, 0)) AS contacts,
    sum(if(has_contact_xwoba, expected_woba, 0)) AS contact_xwoba
  FROM all_pitches
  WHERE pitch_group IS NOT NULL AND zone IS NOT NULL
  GROUP BY ALL
),
league AS (
  SELECT pitch_group, zone, count_group, game_date,
    try_divide(sum(swings) OVER {league_window}, sum(pitches) OVER {league_window})
      AS lg_swing_rate,
    try_divide(sum(whiffs) OVER {league_window}, sum(swings) OVER {league_window})
      AS lg_whiff_rate,
    try_divide(sum(contact_xwoba) OVER {league_window}, sum(contacts) OVER {league_window})
      AS lg_xwoba_contact
  FROM league_days
),
-- Pitches outside a cell with league data (no pitch group or zone) add no chances
pitches AS (
  SELECT p.* EXCEPT (is_swing, is_whiff, is_in_zone, has_contact_xwoba),
    coalesce(p.is_swing AND lg.lg_whiff_rate IS NOT NULL, FALSE) AS is_swing,
    coalesce(p.is_whiff, FALSE) AS is_whiff,
    coalesce(p.is_in_zone OR lg.lg_swing_rate IS NULL, TRUE) AS is_in_zone,
    coalesce(p.has_contact_xwoba AND lg.lg_xwoba_contact IS NOT NULL, FALSE)
      AS has_contact_xwoba,
    lg.lg_swing_rate, lg.lg_whiff_rate, lg.lg_xwoba_contact
  FROM all_pitches p
  LEFT JOIN league lg USING (pitch_group, zone, count_group, game_date)
),
pitch_days AS (
  SELECT
    pitcher_id,
    pitch_type,
    game_date,
    count(*) AS pitches,
    {day_sums}
  FROM pitches
  GROUP BY pitcher_id, pitch_type, game_date
),
-- Every day a pitcher pitched x every pitch type they have thrown, so the windows below
-- have a row to land on even when a pitch type wasn't thrown that day
grid AS (
  SELECT DISTINCT days.pitcher_id, days.game_date, types.pitch_type
  FROM (SELECT DISTINCT pitcher_id, game_date FROM pitch_days) days
  JOIN (SELECT DISTINCT pitcher_id, pitch_type FROM pitch_days) types USING (pitcher_id)
),
grid_days AS (
  SELECT
    g.pitcher_id,
    g.pitch_type,
    g.game_date,
    coalesce(d.pitches, 0) AS pitches,
    {grid_values}
  FROM grid g
  LEFT JOIN pitch_days d USING (pitcher_id, pitch_type, game_date)
),
windows AS (
  SELECT
    g.pitcher_id,
    g.pitch_type,
    g.game_date,
    sum(g.pitches) OVER {window} AS pitches,
    {window_sums}
  FROM grid_days g
),
arsenal AS (
  SELECT
    pitcher_id,
    pitch_type,
    game_date,
    pitches,
    pitches / sum(pitches) OVER (PARTITION BY pitcher_id, game_date) AS usage_rate,
    {averages}
  FROM windows
  WHERE pitches > 0
),
primary_fastball AS (
  SELECT pitcher_id, game_date, pitch_type
  FROM arsenal
  WHERE pitch_type IN ({fastball_types})
  QUALIFY row_number() OVER (
    PARTITION BY pitcher_id, game_date ORDER BY pitches DESC, pitch_type
  ) = 1
)
SELECT
  a.pitcher_id,
  a.pitch_type,
  date_add(a.game_date, 1) AS as_of_date,
  a.game_date AS arsenal_data_through,
  a.pitches AS pitches_thrown,
  a.usage_rate,
  {measures},
  pf.pitch_type AS primary_fastball_type,
  a.pitch_type = pf.pitch_type AS is_primary_fastball,
  {diffs}
FROM arsenal a
LEFT JOIN primary_fastball pf USING (pitcher_id, game_date)
LEFT JOIN arsenal fb
  ON fb.pitcher_id = pf.pitcher_id
  AND fb.game_date = pf.game_date
  AND fb.pitch_type = pf.pitch_type
WHERE {date_filter("a.game_date", source.start, source.end)}
"""
