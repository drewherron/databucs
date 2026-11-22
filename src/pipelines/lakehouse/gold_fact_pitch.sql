-- One row per pitch with everything needed to query it directly: names, teams, game state,
-- pitch and outcome categories, and run value. Built for analysis and model training.
--
-- run_value is the change in run expectancy from the batting team's point of view, using
-- ref_run_expectancy for the same season:
--   runs scored on the pitch + RE(state after) - RE(state before)
-- where the state after the last pitch of a half-inning is worth 0. Positive is good for
-- the hitter. It is NULL on walk-off pitches, where the game ends before the third out and
-- the state after has no meaningful run expectancy.
--
-- savant_run_value (Savant's delta_run_exp) is a different measure: it largely ignores the
-- runners on base (a grand slam is worth about the same as a solo homer, every double play
-- about -0.3 runs). The two agree where runners don't matter and correlate around 0.84
-- overall. run_value is the one that answers "what is this pitch worth in this situation".
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${gold_schema}.fact_pitch (
  CONSTRAINT pitch_key_present
    EXPECT (game_pk IS NOT NULL AND at_bat_number IS NOT NULL AND pitch_number IS NOT NULL)
    ON VIOLATION FAIL UPDATE,
  CONSTRAINT teams_resolved EXPECT (batting_team_id IS NOT NULL AND pitching_team_id IS NOT NULL),
  CONSTRAINT run_value_present EXPECT (run_value IS NOT NULL OR is_walk_off)
)
CLUSTER BY (pitcher_id, batter_id, game_date)
COMMENT 'Pitches with names, teams, game state, outcome flags, and run value. One row per pitch.'
AS
SELECT
  -- Keys and context
  s.game_pk,
  s.at_bat_number,
  s.pitch_number,
  s.game_date,
  s.game_year AS season,
  s.game_type,
  s.inning,
  s.is_top_inning,
  s.batting_team_id,
  s.pitching_team_id,

  -- Matchup
  s.pitcher_id,
  pit.full_name AS pitcher_name,
  s.pitcher_throws,
  s.batter_id,
  bat.full_name AS batter_name,
  s.batter_stands,
  s.pitcher_throws = s.batter_stands AS is_same_side,

  -- State before the pitch
  s.balls,
  s.strikes,
  concat(s.balls, '-', s.strikes) AS count_state,
  s.outs,
  s.base_state,
  s.bat_score - s.fld_score AS bat_score_diff,
  s.times_through_order,
  s.batter_prior_pa_this_game,

  -- Pitch
  s.pitch_type,
  s.pitch_name,
  CASE
    WHEN s.pitch_type IN ('FF', 'SI', 'FC') THEN 'fastball'
    WHEN s.pitch_type IN ('SL', 'ST', 'CU', 'KC', 'SV', 'CS') THEN 'breaking'
    WHEN s.pitch_type IN ('CH', 'FS', 'FO', 'SC') THEN 'offspeed'
    WHEN s.pitch_type IS NOT NULL THEN 'other' -- eephus, knuckleball, pitchout, position players
  END AS pitch_group,
  s.release_speed,
  s.release_spin_rate,
  s.spin_axis,
  s.release_extension,
  s.release_pos_x,
  s.release_pos_z,
  s.arm_angle,
  s.pfx_x,
  s.pfx_z,
  s.plate_x,
  s.plate_z,
  s.zone,
  s.sz_top,
  s.sz_bot,
  s.zone BETWEEN 1 AND 9 AS is_in_zone,

  -- Outcome
  s.description,
  s.events,
  s.description IN ('automatic_ball', 'automatic_strike') AS is_pitch_clock_violation,
  s.description IN (
    'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
    'missed_bunt', 'bunt_foul_tip', 'hit_into_play', 'swinging_pitchout'
  ) AS is_swing,
  s.description IN (
    'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'swinging_pitchout'
  ) AS is_whiff,
  s.description = 'called_strike' AS is_called_strike,
  s.description = 'hit_into_play' AS is_in_play,
  s.events IS NOT NULL AS is_final_pitch_of_pa,
  s.is_last_pitch_of_half_inning,
  s.is_walk_off,
  s.batted_ball_type,
  s.launch_speed,
  s.launch_angle,
  s.expected_woba,
  s.woba_value,
  s.woba_denom,
  s.bat_speed,
  s.swing_length,

  -- Run value
  s.runs_on_pitch,
  re_before.run_expectancy AS run_expectancy_before,
  -- The last pitch of a half-inning leaves nothing to expect
  if(s.is_last_pitch_of_half_inning, 0, re_after.run_expectancy) AS run_expectancy_after,
  if(
    s.is_walk_off,
    NULL,
    s.runs_on_pitch + if(s.is_last_pitch_of_half_inning, 0, re_after.run_expectancy)
      - re_before.run_expectancy
  ) AS run_value,
  s.delta_run_exp AS savant_run_value
FROM pitch_states s
LEFT JOIN ${catalog}.${silver_schema}.players pit ON pit.player_id = s.pitcher_id
LEFT JOIN ${catalog}.${silver_schema}.players bat ON bat.player_id = s.batter_id
LEFT JOIN ${catalog}.${gold_schema}.ref_run_expectancy re_before
  ON re_before.season = s.game_year
  AND re_before.outs = s.outs
  AND re_before.base_state = s.base_state
  AND re_before.balls = s.balls
  AND re_before.strikes = s.strikes
LEFT JOIN ${catalog}.${gold_schema}.ref_run_expectancy re_after
  ON re_after.season = s.game_year
  AND re_after.outs = s.next_outs
  AND re_after.base_state = s.next_base_state
  AND re_after.balls = s.next_balls
  AND re_after.strikes = s.next_strikes;
