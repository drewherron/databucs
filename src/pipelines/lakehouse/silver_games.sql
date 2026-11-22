-- One row per MLB game (regular season and postseason), from the schedule responses.
-- See schedule_games (silver_statsapi_views.sql) for how a game listed more than once is
-- resolved to a single entry.
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
SELECT
  game_pk,
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
FROM schedule_games;
