"""game_state_features: the situation before each pitch.

One row per pitch, keyed on (game_pk, at_bat_number, pitch_number). Everything here is
known before the pitch is thrown, so there is no time key: the row for a pitch is always
safe to use for that pitch. When the recommender scores a hypothetical situation, it passes
these values in directly instead of looking them up.
"""

from __future__ import annotations

from databucs.features.common import Source, date_filter

TABLE = "game_state_features"
PRIMARY_KEYS = ["game_pk", "at_bat_number", "pitch_number"]
TIMESERIES_COLUMN = None

# Late and close (the Elias definition): seventh inning or later, with the batting team
# ahead by one, tied, or with the tying run on base, at bat or on deck
LATE_INNING = 7


def build_sql(source: Source) -> str:
    """SQL for the game state rows for game days in [source.start, source.end]."""
    return f"""
SELECT
  game_pk,
  at_bat_number,
  pitch_number,
  balls,
  strikes,
  count_state,
  CASE
    WHEN strikes = 2 THEN 'two_strikes'
    WHEN balls > strikes THEN 'hitter_ahead'
    ELSE 'neutral'
  END AS count_group,
  outs,
  base_state,
  base_state <> '___' AS runners_on,
  base_state LIKE '%2%' OR base_state LIKE '%3%' AS runner_in_scoring_position,
  inning,
  is_top_inning,
  bat_score_diff,
  inning >= {LATE_INNING}
    AND bat_score_diff <= 1
    -- runners on base + the batter + the on-deck hitter can tie it
    AND -bat_score_diff <= length(replace(base_state, '_', '')) + 2 AS is_late_and_close,
  times_through_order,
  pitch_number = 1 AS is_first_pitch_of_pa,
  is_same_side AS is_platoon_same_side
FROM {source.fact_pitch}
WHERE {date_filter("game_date", source.start, source.end)}
"""
