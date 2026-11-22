-- Cleaned, typed Statcast pitches: one row per pitch, keyed on
-- (game_pk, at_bat_number, pitch_number).
--
-- Bronze can hold more than one version of a day when a corrected re-pull lands as a new
-- file. Each day is published as a single file, so the latest file for a game date replaces
-- all earlier ones; that also drops any pitch the correction removed.
--
-- Values are cast with try_cast so a malformed value becomes NULL instead of failing the
-- whole update; the expectations below and the data quality checks catch those rows.
--
-- Dropped from bronze: columns Savant marks deprecated or leaves empty (spin_dir,
-- *_deprecated, umpire, sv_id), the legacy age columns, and player_name (the pitcher's name;
-- names come from the players table).
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.pitches (
  CONSTRAINT pitch_key_present
    EXPECT (game_pk IS NOT NULL AND at_bat_number IS NOT NULL AND pitch_number IS NOT NULL)
    ON VIOLATION FAIL UPDATE,
  CONSTRAINT players_present EXPECT (pitcher_id IS NOT NULL AND batter_id IS NOT NULL),
  CONSTRAINT valid_count EXPECT (balls BETWEEN 0 AND 3 AND strikes BETWEEN 0 AND 2),
  CONSTRAINT valid_outs EXPECT (outs BETWEEN 0 AND 2)
)
CLUSTER BY (game_date, pitcher_id)
COMMENT 'Statcast pitches, typed and deduplicated. One row per pitch.'
AS
WITH latest AS (
  SELECT *
  FROM statcast_pitches
  QUALIFY _source_file_modified_at
    = max(_source_file_modified_at) OVER (PARTITION BY game_date)
)
SELECT
  -- Keys and game context
  try_cast(game_pk AS INT) AS game_pk,
  try_cast(at_bat_number AS INT) AS at_bat_number,
  try_cast(pitch_number AS INT) AS pitch_number,
  try_cast(game_date AS DATE) AS game_date,
  try_cast(game_year AS INT) AS game_year,
  game_type,
  home_team,
  away_team,
  try_cast(inning AS INT) AS inning,
  inning_topbot = 'Top' AS is_top_inning,

  -- Players
  try_cast(pitcher AS INT) AS pitcher_id,
  try_cast(batter AS INT) AS batter_id,
  p_throws AS pitcher_throws,
  stand AS batter_stands,
  try_cast(age_pit AS INT) AS pitcher_age,
  try_cast(age_bat AS INT) AS batter_age,
  try_cast(fielder_2 AS INT) AS catcher_id,
  try_cast(fielder_3 AS INT) AS first_base_id,
  try_cast(fielder_4 AS INT) AS second_base_id,
  try_cast(fielder_5 AS INT) AS third_base_id,
  try_cast(fielder_6 AS INT) AS shortstop_id,
  try_cast(fielder_7 AS INT) AS left_field_id,
  try_cast(fielder_8 AS INT) AS center_field_id,
  try_cast(fielder_9 AS INT) AS right_field_id,

  -- State before the pitch
  try_cast(balls AS INT) AS balls,
  try_cast(strikes AS INT) AS strikes,
  try_cast(outs_when_up AS INT) AS outs,
  try_cast(on_1b AS INT) AS runner_on_1b_id,
  try_cast(on_2b AS INT) AS runner_on_2b_id,
  try_cast(on_3b AS INT) AS runner_on_3b_id,
  try_cast(home_score AS INT) AS home_score,
  try_cast(away_score AS INT) AS away_score,
  try_cast(bat_score AS INT) AS bat_score,
  try_cast(fld_score AS INT) AS fld_score,
  try_cast(n_thruorder_pitcher AS INT) AS times_through_order,
  try_cast(n_priorpa_thisgame_player_at_bat AS INT) AS batter_prior_pa_this_game,
  try_cast(pitcher_days_since_prev_game AS INT) AS pitcher_days_since_prev_game,
  try_cast(batter_days_since_prev_game AS INT) AS batter_days_since_prev_game,
  if_fielding_alignment AS infield_alignment,
  of_fielding_alignment AS outfield_alignment,
  try_cast(home_win_exp AS DOUBLE) AS home_win_exp,
  try_cast(bat_win_exp AS DOUBLE) AS bat_win_exp,

  -- Pitch identity
  pitch_type,
  pitch_name,

  -- Release
  try_cast(release_speed AS DOUBLE) AS release_speed,
  try_cast(effective_speed AS DOUBLE) AS effective_speed,
  try_cast(release_spin_rate AS INT) AS release_spin_rate,
  try_cast(spin_axis AS INT) AS spin_axis,
  try_cast(release_extension AS DOUBLE) AS release_extension,
  try_cast(release_pos_x AS DOUBLE) AS release_pos_x,
  try_cast(release_pos_y AS DOUBLE) AS release_pos_y,
  try_cast(release_pos_z AS DOUBLE) AS release_pos_z,
  try_cast(arm_angle AS DOUBLE) AS arm_angle,

  -- Movement and trajectory
  try_cast(pfx_x AS DOUBLE) AS pfx_x,
  try_cast(pfx_z AS DOUBLE) AS pfx_z,
  try_cast(api_break_z_with_gravity AS DOUBLE) AS break_z_with_gravity,
  try_cast(api_break_x_arm AS DOUBLE) AS break_x_arm,
  try_cast(api_break_x_batter_in AS DOUBLE) AS break_x_batter_in,
  try_cast(vx0 AS DOUBLE) AS vx0,
  try_cast(vy0 AS DOUBLE) AS vy0,
  try_cast(vz0 AS DOUBLE) AS vz0,
  try_cast(ax AS DOUBLE) AS ax,
  try_cast(ay AS DOUBLE) AS ay,
  try_cast(az AS DOUBLE) AS az,

  -- Location
  try_cast(plate_x AS DOUBLE) AS plate_x,
  try_cast(plate_z AS DOUBLE) AS plate_z,
  try_cast(zone AS INT) AS zone,
  try_cast(sz_top AS DOUBLE) AS sz_top,
  try_cast(sz_bot AS DOUBLE) AS sz_bot,

  -- Result
  type AS result_type,
  description,
  events,
  des AS event_description,
  bb_type AS batted_ball_type,
  try_cast(hit_location AS INT) AS hit_location,
  try_cast(launch_speed AS DOUBLE) AS launch_speed,
  try_cast(launch_angle AS DOUBLE) AS launch_angle,
  try_cast(launch_speed_angle AS INT) AS launch_speed_angle,
  try_cast(hyper_speed AS DOUBLE) AS hyper_speed,
  try_cast(hit_distance_sc AS INT) AS hit_distance,
  try_cast(hc_x AS DOUBLE) AS hit_coord_x,
  try_cast(hc_y AS DOUBLE) AS hit_coord_y,
  try_cast(estimated_ba_using_speedangle AS DOUBLE) AS expected_ba,
  try_cast(estimated_woba_using_speedangle AS DOUBLE) AS expected_woba,
  try_cast(estimated_slg_using_speedangle AS DOUBLE) AS expected_slg,
  try_cast(woba_value AS DOUBLE) AS woba_value,
  try_cast(woba_denom AS INT) AS woba_denom,
  try_cast(babip_value AS INT) AS babip_value,
  try_cast(iso_value AS INT) AS iso_value,

  -- Swing (bat tracking, 2024 on)
  try_cast(bat_speed AS DOUBLE) AS bat_speed,
  try_cast(swing_length AS DOUBLE) AS swing_length,
  try_cast(miss_distance AS DOUBLE) AS miss_distance,
  try_cast(attack_angle AS DOUBLE) AS attack_angle,
  try_cast(attack_direction AS DOUBLE) AS attack_direction,
  try_cast(swing_path_tilt AS DOUBLE) AS swing_path_tilt,
  try_cast(intercept_ball_minus_batter_pos_x_inches AS DOUBLE)
    AS intercept_ball_minus_batter_pos_x_inches,
  try_cast(intercept_ball_minus_batter_pos_y_inches AS DOUBLE)
    AS intercept_ball_minus_batter_pos_y_inches,

  -- State after the pitch and value
  try_cast(post_home_score AS INT) AS post_home_score,
  try_cast(post_away_score AS INT) AS post_away_score,
  try_cast(post_bat_score AS INT) AS post_bat_score,
  try_cast(post_fld_score AS INT) AS post_fld_score,
  try_cast(delta_run_exp AS DOUBLE) AS delta_run_exp,
  try_cast(delta_pitcher_run_exp AS DOUBLE) AS delta_pitcher_run_exp,
  try_cast(delta_home_win_exp AS DOUBLE) AS delta_home_win_exp,

  -- Lineage
  _source_file,
  _source_file_modified_at
FROM latest;
