-- One row per MLB game (regular season and postseason), from the schedule responses.
--
-- A game can appear in the schedule more than once:
--   - a postponed game appears on its original date (status D) and again on its make-up date;
--   - a suspended or rain-shortened game that was resumed appears on both dates;
--   - re-pulled schedule files hold another copy of the same games.
-- Only the latest version of each schedule file is used. Of a game's appearances, the one
-- that was played is kept, then the latest scheduled date, then the newest file.
--
-- official_date is the date the game counts for. Statcast files all of a game's pitches
-- under it, even when part of the game was played later, so it is the date to join on.
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.games (
  CONSTRAINT game_pk_present EXPECT (game_pk IS NOT NULL) ON VIOLATION FAIL UPDATE,
  CONSTRAINT teams_present EXPECT (home_team_id IS NOT NULL AND away_team_id IS NOT NULL),
  CONSTRAINT played_games_have_scores
    EXPECT (NOT is_played OR (home_score IS NOT NULL AND away_score IS NOT NULL))
)
CLUSTER BY (official_date)
COMMENT 'MLB games from the Stats API schedule. One row per game.'
AS
WITH latest_files AS (
  SELECT payload, _source_file, _source_file_modified_at
  FROM statsapi_responses
  WHERE endpoint = 'schedule'
  QUALIFY _source_file_modified_at = max(_source_file_modified_at) OVER (
    -- Files of the same request share a name up to the content hash.
    PARTITION BY regexp_extract(_source_file, '/(schedule_[0-9-]+_[0-9-]+)_[0-9a-f]+\\.json$', 1)
  )
),
appearances AS (
  SELECT
    g.value AS game,
    g.value:status.abstractGameState::STRING = 'Final'
      AND g.value:status.codedGameState::STRING NOT IN ('D', 'C') AS is_played,
    f._source_file,
    f._source_file_modified_at
  FROM latest_files f,
    LATERAL variant_explode(f.payload:dates) AS d,
    LATERAL variant_explode(d.value:games) AS g
),
chosen AS (
  SELECT *
  FROM appearances
  QUALIFY row_number() OVER (
    PARTITION BY game:gamePk::INT
    ORDER BY is_played DESC, game:gameDate::TIMESTAMP DESC, _source_file_modified_at DESC
  ) = 1
)
SELECT
  game:gamePk::INT AS game_pk,
  game:season::INT AS season,
  game:gameType::STRING AS game_type,
  game:seriesDescription::STRING AS series_description,
  game:officialDate::DATE AS official_date,
  game:gameDate::TIMESTAMP AS start_time,
  game:dayNight::STRING AS day_night,
  game:doubleHeader::STRING AS double_header,
  game:gameNumber::INT AS game_number,
  game:scheduledInnings::INT AS scheduled_innings,
  game:seriesGameNumber::INT AS series_game_number,
  game:gamesInSeries::INT AS games_in_series,

  -- Status
  is_played,
  game:status.codedGameState::STRING AS status_code,
  game:status.detailedState::STRING AS status,
  game:status.reason::STRING AS status_reason,
  game:resumedFromDate::DATE AS resumed_from_date,

  -- Teams and result
  game:teams.home.team.id::INT AS home_team_id,
  game:teams.home.team.name::STRING AS home_team_name,
  game:teams.away.team.id::INT AS away_team_id,
  game:teams.away.team.name::STRING AS away_team_name,
  game:teams.home.score::INT AS home_score,
  game:teams.away.score::INT AS away_score,
  game:teams.home.isWinner::BOOLEAN AS home_won,
  game:isTie::BOOLEAN AS is_tie,

  -- Probable starters as announced before the game (not necessarily who started)
  game:teams.home.probablePitcher.id::INT AS home_probable_pitcher_id,
  game:teams.away.probablePitcher.id::INT AS away_probable_pitcher_id,

  game:venue.id::INT AS venue_id,
  game:venue.name::STRING AS venue_name,

  -- Lineage
  _source_file,
  _source_file_modified_at
FROM chosen;
