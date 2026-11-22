-- Game state around each pitch, shared by the gold tables (pipeline-internal, not published).
--
-- Statcast records the state *before* each pitch (count, outs, runners, score). The state
-- after a pitch is the state before the next pitch in the same half-inning; after the last
-- pitch of a half-inning there is no next state.
--
-- A walk-off ends the game before the third out, so the state after a walk-off pitch has
-- no run expectancy to speak of; those pitches are flagged.
CREATE TEMPORARY VIEW pitch_states
COMMENT 'Silver pitches with team IDs and the game state before and after each pitch.'
AS
WITH base AS (
  SELECT
    p.*,
    if(p.is_top_inning, g.away_team_id, g.home_team_id) AS batting_team_id,
    if(p.is_top_inning, g.home_team_id, g.away_team_id) AS pitching_team_id,
    -- Runners as a 3-character code: '1_3' means runners on first and third
    concat(
      if(p.runner_on_1b_id IS NULL, '_', '1'),
      if(p.runner_on_2b_id IS NULL, '_', '2'),
      if(p.runner_on_3b_id IS NULL, '_', '3')
    ) AS base_state
  FROM ${catalog}.${silver_schema}.pitches p
  JOIN ${catalog}.${silver_schema}.games g USING (game_pk)
),
states AS (
  SELECT
    *,
    post_bat_score - bat_score AS runs_on_pitch,
    (max(post_bat_score) OVER half_inning) - bat_score AS runs_to_end_of_inning,
    lead(outs) OVER half_inning_order AS next_outs,
    lead(base_state) OVER half_inning_order AS next_base_state,
    lead(balls) OVER half_inning_order AS next_balls,
    lead(strikes) OVER half_inning_order AS next_strikes
  FROM base
  WINDOW
    half_inning AS (PARTITION BY game_pk, inning, is_top_inning),
    half_inning_order AS (
      PARTITION BY game_pk, inning, is_top_inning ORDER BY at_bat_number, pitch_number
    )
)
SELECT
  *,
  next_outs IS NULL AS is_last_pitch_of_half_inning,
  -- The home team only bats in the ninth or later while not ahead, so finishing the bottom
  -- half ahead means the game ended on that pitch
  next_outs IS NULL AND NOT is_top_inning AND inning >= 9
    AND post_bat_score > post_fld_score AS is_walk_off
FROM states;
