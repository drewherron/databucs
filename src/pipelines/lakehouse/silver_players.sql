-- MLB players: one row per player who appeared in a season we've fetched. Attributes come
-- from the latest season the player appears in (handedness, size, and names rarely change).
--
-- The source is the season's player list, which doesn't record which team a player was on
-- in that season (currentTeam reflects the date of the pull), so team membership comes from
-- games and lineups instead.
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.players (
  CONSTRAINT player_id_present EXPECT (player_id IS NOT NULL) ON VIOLATION FAIL UPDATE,
  CONSTRAINT handedness_known EXPECT (bats IS NOT NULL AND throws IS NOT NULL)
)
COMMENT 'MLB players. One row per player.'
AS
WITH by_season AS (
  SELECT
    p.value AS player,
    try_cast(regexp_extract(s.request, '^players_([0-9]{4})$', 1) AS INT) AS season,
    s._source_file,
    s._source_file_modified_at
  FROM statsapi_latest s,
    LATERAL variant_explode(s.payload:people) AS p
  WHERE s.endpoint = 'players'
)
SELECT
  player:id::INT AS player_id,
  player:fullName::STRING AS full_name,
  player:firstName::STRING AS first_name,
  player:lastName::STRING AS last_name,
  player:useName::STRING AS use_name,
  player:boxscoreName::STRING AS boxscore_name,
  player:batSide.code::STRING AS bats,
  player:pitchHand.code::STRING AS throws,
  player:primaryPosition.abbreviation::STRING AS primary_position,
  player:primaryPosition.type::STRING AS position_type,
  player:birthDate::DATE AS birth_date,
  player:birthCountry::STRING AS birth_country,
  -- Height arrives as text like 6' 2"
  try_cast(regexp_extract(player:height::STRING, '^(\\d+)\'', 1) AS INT) * 12
    + try_cast(regexp_extract(player:height::STRING, '(\\d+)"', 1) AS INT) AS height_inches,
  player:weight::INT AS weight_lbs,
  player:mlbDebutDate::DATE AS mlb_debut_date,
  -- Personal strike zone, in feet; Statcast's sz_top/sz_bot are per pitch
  player:strikeZoneTop::DOUBLE AS strike_zone_top,
  player:strikeZoneBottom::DOUBLE AS strike_zone_bottom,
  min(season) OVER (PARTITION BY player:id::INT) AS first_season_in_data,
  season AS last_season_in_data,
  _source_file,
  _source_file_modified_at
FROM by_season
QUALIFY row_number() OVER (PARTITION BY player:id::INT ORDER BY season DESC) = 1;
