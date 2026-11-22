-- Run expectancy: the average number of runs the batting team scores from a given state
-- until the end of the half-inning, by season, outs, runners, and count.
--
-- The 0-0 count rows are the classic 24-state run expectancy (RE24) table, since every
-- plate appearance starts at 0-0. The other counts give the run value of the count itself.
--
-- Computed from regular-season innings 1-8 only, the usual convention: a walk-off ends the
-- bottom of the ninth or extra innings early, and extra innings start with a runner on
-- second. States seen rarely (bases loaded at 3-0, say) have small samples, so sample_size
-- is kept for downstream smoothing.
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${gold_schema}.ref_run_expectancy (
  CONSTRAINT valid_state
    EXPECT (outs BETWEEN 0 AND 2 AND balls BETWEEN 0 AND 3 AND strikes BETWEEN 0 AND 2)
)
COMMENT 'Average runs to the end of the half-inning by season, outs, runners, and count.'
AS
SELECT
  game_year AS season,
  outs,
  base_state,
  balls,
  strikes,
  avg(runs_to_end_of_inning) AS run_expectancy,
  count(*) AS sample_size
FROM pitch_states
WHERE game_type = 'R'
  AND inning <= 8
GROUP BY ALL;
