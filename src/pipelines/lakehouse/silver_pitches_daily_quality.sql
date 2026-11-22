-- Daily completeness check for Statcast pitches: one row per date with played games or
-- quarantined pitches. It compares the schedule's played games with the pitches that
-- reached silver.pitches, and lists anything abnormal in `issues`:
--
--   games_missing_pitches   a played game has no pitches (Savant hasn't posted it, or the
--                           day's file was never fetched)
--   game_with_few_pitches   a game has fewer than 100 pitches. Complete games have at least
--                           about 200; the fewest in 2023-2026 was 143, in a game called
--                           after five innings. Fewer usually means a partial file.
--   pitches_rejected        some pitches failed a row rule (see pitches_quarantine)
--
-- The check is per game rather than a daily total compared with other days, because the
-- number of games varies a lot from day to day while pitches per game hardly does.
-- Measurement values set to NULL are counted but not an issue: a few per week is normal.
-- The ingest job's check_data_quality task fails the run when a recent day has issues.
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.pitches_daily_quality
COMMENT 'Per-day pitch completeness and data quality results, with any issues listed.'
AS
WITH game_pitches AS (
  SELECT g.official_date AS game_date, g.game_pk, count(p.game_pk) AS pitches
  FROM ${catalog}.${silver_schema}.games g
  LEFT JOIN ${catalog}.${silver_schema}.pitches p USING (game_pk)
  WHERE g.is_played
  GROUP BY ALL
),
games_by_day AS (
  SELECT
    game_date,
    count(*) AS games_played,
    count_if(pitches > 0) AS games_with_pitches,
    sum(pitches) AS pitches,
    min(pitches) FILTER (WHERE pitches > 0) AS fewest_pitches_in_a_game
  FROM game_pitches
  GROUP BY game_date
),
quarantine_by_day AS (
  SELECT
    game_date,
    count_if(action = 'rejected') AS pitches_rejected,
    count_if(action = 'values_nulled') AS pitches_with_values_nulled
  FROM ${catalog}.${silver_schema}.pitches_quarantine
  GROUP BY game_date
),
combined AS (
  SELECT
    coalesce(g.game_date, q.game_date) AS game_date,
    coalesce(g.games_played, 0) AS games_played,
    coalesce(g.games_with_pitches, 0) AS games_with_pitches,
    coalesce(g.pitches, 0) AS pitches,
    g.fewest_pitches_in_a_game,
    coalesce(q.pitches_rejected, 0) AS pitches_rejected,
    coalesce(q.pitches_with_values_nulled, 0) AS pitches_with_values_nulled
  FROM games_by_day g
  FULL OUTER JOIN quarantine_by_day q ON q.game_date = g.game_date
)
SELECT
  *,
  filter(
    array(
      if(games_with_pitches < games_played, 'games_missing_pitches', NULL),
      if(fewest_pitches_in_a_game < 100, 'game_with_few_pitches', NULL),
      if(pitches_rejected > 0, 'pitches_rejected', NULL)
    ),
    x -> x IS NOT NULL
  ) AS issues
FROM combined;
