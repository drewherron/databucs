-- Pipeline-internal views shared by the silver tables built from Stats API responses.
-- Temporary views are not published to Unity Catalog.

-- The latest version of each landed response. A re-pull of the same request with changed
-- content lands as a new file whose name differs only in the content hash, e.g.
-- players_2025_3de734fd8e26.json and players_2025_98491b65c4f5.json.
CREATE TEMPORARY VIEW statsapi_latest
COMMENT 'Latest landed version of each Stats API request.'
AS
SELECT
  endpoint,
  regexp_extract(_source_file, '/([^/]+)_[0-9a-f]+\\.json$', 1) AS request,
  payload,
  _source_file,
  _source_file_modified_at
FROM statsapi_responses
QUALIFY _source_file_modified_at = max(_source_file_modified_at) OVER (
  PARTITION BY regexp_extract(_source_file, '/([^/]+)_[0-9a-f]+\\.json$', 1)
);

-- One schedule entry per game. A game can be listed more than once:
--   - a postponed game on its original date (status D) and again on its make-up date;
--   - a suspended or rain-shortened game that was resumed, on both dates;
--   - the same game in overlapping schedule requests.
-- The entry for the played game is kept, then the latest scheduled date, then the newest file.
CREATE TEMPORARY VIEW schedule_games
COMMENT 'One schedule entry (raw VARIANT) per game.'
AS
WITH entries AS (
  SELECT
    g.value:gamePk::INT AS game_pk,
    g.value AS game,
    g.value:status.abstractGameState::STRING = 'Final'
      AND g.value:status.codedGameState::STRING NOT IN ('D', 'C') AS is_played,
    s._source_file,
    s._source_file_modified_at
  FROM statsapi_latest s,
    LATERAL variant_explode(s.payload:dates) AS d,
    LATERAL variant_explode(d.value:games) AS g
  WHERE s.endpoint = 'schedule'
)
SELECT *
FROM entries
QUALIFY row_number() OVER (
  PARTITION BY game_pk
  ORDER BY is_played DESC, game:gameDate::TIMESTAMP DESC, _source_file_modified_at DESC
) = 1;
