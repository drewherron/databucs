-- Typed Statcast pitches with the data quality rules applied (pipeline-internal, not
-- published). silver.pitches and silver.pitches_quarantine both read from this view.
--
-- Bronze can hold more than one version of a day when a corrected re-pull lands as a new
-- file. Each day is published as a single file, so the latest file for a game date replaces
-- all earlier ones; that also drops any pitch the correction removed.
--
-- Values are cast with try_cast so a malformed value becomes NULL instead of failing the
-- whole update. A missing key still fails the update.
--
-- Every rule is a field of `checks`, TRUE when the pitch passes. There are two kinds:
--
-- * Row rules: the pitch itself can't be trusted (missing players, impossible count or
--   outs, no matching played game, duplicate key, a gap in its plate appearance's pitch
--   numbers). These pitches are rejected and left out of silver.pitches.
-- * Value rules: a tracking measurement is physically implausible, such as a pitch crossing
--   the plate 57 feet underground or a 94 mph sinker with 81 rpm of spin. The pitch is kept,
--   since it was thrown and its count and outcome are fine, but the failing group of
--   measurements is set to NULL, the same as an ordinary tracking gap.
--
-- The ranges are wide on purpose. They reject readings that can't be real, not unusual
-- pitches: position-player lobs (30-40 mph, 3 ft extension), Tyler Rogers' submarine
-- release (under 1 ft high), knuckleballs and eephus pitches with almost no spin, and
-- spiked or wild pitches that cross the plate far out of the zone all pass.
--
-- Each rule is also an expectation, so the pipeline event log records pass and fail counts
-- per rule on every update. Pitches that fail any rule go to silver.pitches_quarantine.
CREATE TEMPORARY VIEW pitch_checks (
  CONSTRAINT pitch_key_present EXPECT (checks.pitch_key_present) ON VIOLATION FAIL UPDATE,
  -- Row rules
  CONSTRAINT players_present EXPECT (checks.players_present),
  CONSTRAINT valid_count EXPECT (checks.valid_count),
  CONSTRAINT valid_outs EXPECT (checks.valid_outs),
  CONSTRAINT known_game EXPECT (checks.known_game),
  CONSTRAINT unique_pitch_key EXPECT (checks.unique_pitch_key),
  CONSTRAINT complete_plate_appearance EXPECT (checks.complete_plate_appearance),
  -- Value rules
  CONSTRAINT plausible_velocity EXPECT (checks.plausible_velocity),
  CONSTRAINT plausible_spin EXPECT (checks.plausible_spin),
  CONSTRAINT plausible_location EXPECT (checks.plausible_location),
  CONSTRAINT plausible_release EXPECT (checks.plausible_release),
  CONSTRAINT plausible_strike_zone EXPECT (checks.plausible_strike_zone),
  CONSTRAINT known_pitch_type EXPECT (checks.known_pitch_type)
)
COMMENT 'Typed Statcast pitches with data quality rule results.'
AS
WITH latest AS (
  SELECT *
  FROM statcast_pitches
  QUALIFY _source_file_modified_at
    = max(_source_file_modified_at) OVER (PARTITION BY game_date)
),

-- The values the rules look at, typed. Kept as a struct so the quarantine table can show
-- the original values even where silver.pitches nulls them.
typed AS (
  SELECT
    *,
    struct(
      try_cast(game_pk AS INT) AS game_pk,
      try_cast(at_bat_number AS INT) AS at_bat_number,
      try_cast(pitch_number AS INT) AS pitch_number,
      try_cast(game_date AS DATE) AS game_date,
      try_cast(pitcher AS INT) AS pitcher_id,
      try_cast(batter AS INT) AS batter_id,
      try_cast(balls AS INT) AS balls,
      try_cast(strikes AS INT) AS strikes,
      try_cast(outs_when_up AS INT) AS outs,
      pitch_type,
      try_cast(release_speed AS DOUBLE) AS release_speed,
      try_cast(effective_speed AS DOUBLE) AS effective_speed,
      try_cast(release_spin_rate AS INT) AS release_spin_rate,
      try_cast(spin_axis AS INT) AS spin_axis,
      try_cast(plate_x AS DOUBLE) AS plate_x,
      try_cast(plate_z AS DOUBLE) AS plate_z,
      try_cast(release_pos_x AS DOUBLE) AS release_pos_x,
      try_cast(release_pos_z AS DOUBLE) AS release_pos_z,
      try_cast(release_extension AS DOUBLE) AS release_extension,
      try_cast(sz_top AS DOUBLE) AS sz_top,
      try_cast(sz_bot AS DOUBLE) AS sz_bot
    ) AS v
  FROM latest
),

with_context AS (
  SELECT
    t.*,
    g.is_played AND g.official_date = t.v.game_date AS game_matches,
    count(*) OVER pitch_key AS pitch_key_copies,
    min(t.v.pitch_number) OVER plate_appearance AS pa_first_pitch,
    max(t.v.pitch_number) OVER plate_appearance AS pa_last_pitch,
    count(*) OVER plate_appearance AS pa_pitches
  FROM typed t
  LEFT JOIN ${catalog}.${silver_schema}.games g ON g.game_pk = t.v.game_pk
  WINDOW
    pitch_key AS (PARTITION BY t.v.game_pk, t.v.at_bat_number, t.v.pitch_number),
    plate_appearance AS (PARTITION BY t.v.game_pk, t.v.at_bat_number)
),

checked AS (
  SELECT
    *,
    named_struct(
      'pitch_key_present',
        v.game_pk IS NOT NULL AND v.at_bat_number IS NOT NULL AND v.pitch_number IS NOT NULL,

      -- Row rules
      'players_present', v.pitcher_id IS NOT NULL AND v.batter_id IS NOT NULL,
      'valid_count', coalesce(v.balls BETWEEN 0 AND 3 AND v.strikes BETWEEN 0 AND 2, false),
      'valid_outs', coalesce(v.outs BETWEEN 0 AND 2, false),
      -- A played game in silver.games, filed under the game's official date
      'known_game', coalesce(game_matches, false),
      'unique_pitch_key', pitch_key_copies = 1,
      -- Pitch numbers run 1..n with no gaps; otherwise a pitch is missing and the count and
      -- state sequence of the whole plate appearance can't be trusted
      'complete_plate_appearance',
        coalesce(pa_first_pitch = 1 AND pa_last_pitch = pa_pitches, false),

      -- Value rules (NULL is a tracking gap, not a failure)
      'plausible_velocity',
        (v.release_speed IS NULL OR v.release_speed BETWEEN 25 AND 110)
        AND (v.effective_speed IS NULL OR v.effective_speed BETWEEN 25 AND 110),
      -- Under 300 rpm is a misread, except on knuckleballs, eephus pitches and position
      -- players' lobs (FA), which really can have almost no spin
      'plausible_spin',
        (v.release_spin_rate IS NULL
          OR v.release_spin_rate BETWEEN 300 AND 4000
          OR (v.release_spin_rate BETWEEN 0 AND 299
            AND coalesce(v.pitch_type IN ('KN', 'EP', 'FA'), false)))
        AND (v.spin_axis IS NULL OR v.spin_axis BETWEEN 0 AND 360),
      -- In feet from the middle of the plate and above the ground. Spiked pitches have a
      -- negative plate_z (the trajectory is extrapolated through the bounce).
      'plausible_location',
        (v.plate_x IS NULL OR abs(v.plate_x) <= 10)
        AND (v.plate_z IS NULL OR v.plate_z BETWEEN -10 AND 15),
      'plausible_release',
        (v.release_pos_x IS NULL OR abs(v.release_pos_x) <= 6)
        AND (v.release_pos_z IS NULL OR v.release_pos_z BETWEEN 0.3 AND 9)
        AND (v.release_extension IS NULL OR v.release_extension BETWEEN 2.5 AND 9.5),
      'plausible_strike_zone',
        v.sz_top IS NULL OR v.sz_bot IS NULL
        OR (v.sz_bot BETWEEN 0.5 AND 3 AND v.sz_top BETWEEN 1.5 AND 5 AND v.sz_bot < v.sz_top),
      'known_pitch_type',
        v.pitch_type IS NULL
        OR v.pitch_type IN (
          'FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'KC', 'CS', 'CH', 'FS', 'FO', 'SC', 'KN',
          'EP', 'FA', 'PO', 'UN'
        )
    ) AS checks
  FROM with_context
)

SELECT
  -- Keys and game context
  v.game_pk,
  v.at_bat_number,
  v.pitch_number,
  v.game_date,
  try_cast(game_year AS INT) AS game_year,
  game_type,
  home_team,
  away_team,
  try_cast(inning AS INT) AS inning,
  inning_topbot = 'Top' AS is_top_inning,

  -- Players
  v.pitcher_id,
  v.batter_id,
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
  v.balls,
  v.strikes,
  v.outs,
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
  if(checks.known_pitch_type, v.pitch_type, NULL) AS pitch_type,
  if(checks.known_pitch_type, pitch_name, NULL) AS pitch_name,

  -- Release
  if(checks.plausible_velocity, v.release_speed, NULL) AS release_speed,
  if(checks.plausible_velocity, v.effective_speed, NULL) AS effective_speed,
  if(checks.plausible_spin, v.release_spin_rate, NULL) AS release_spin_rate,
  if(checks.plausible_spin, v.spin_axis, NULL) AS spin_axis,
  if(checks.plausible_release, v.release_extension, NULL) AS release_extension,
  if(checks.plausible_release, v.release_pos_x, NULL) AS release_pos_x,
  if(checks.plausible_release, try_cast(release_pos_y AS DOUBLE), NULL) AS release_pos_y,
  if(checks.plausible_release, v.release_pos_z, NULL) AS release_pos_z,
  if(checks.plausible_release, try_cast(arm_angle AS DOUBLE), NULL) AS arm_angle,

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
  if(checks.plausible_location, v.plate_x, NULL) AS plate_x,
  if(checks.plausible_location, v.plate_z, NULL) AS plate_z,
  if(checks.plausible_location, try_cast(zone AS INT), NULL) AS zone,
  if(checks.plausible_strike_zone, v.sz_top, NULL) AS sz_top,
  if(checks.plausible_strike_zone, v.sz_bot, NULL) AS sz_bot,

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
  _source_file_modified_at,

  -- Data quality
  checks,
  -- Names of the rules this pitch failed, empty when it passed them all
  filter(
    transform(map_entries(from_json(to_json(checks), 'MAP<STRING, BOOLEAN>')),
      e -> if(e.value, NULL, e.key)),
    x -> x IS NOT NULL
  ) AS failed_rules,
  NOT (
    checks.players_present AND checks.valid_count AND checks.valid_outs AND checks.known_game
    AND checks.unique_pitch_key AND checks.complete_plate_appearance
  ) AS is_rejected,
  v AS checked_values
FROM checked;
