-- MLB teams by season: one row per (team_id, season). Names, abbreviations, league, and
-- division can change between seasons (the Athletics were OAK through 2024, ATH since).
--
-- Join pitches to teams through games (home_team_id / away_team_id), not by abbreviation:
-- Statcast relabels past seasons with a team's current abbreviation (ATH in 2023), while
-- the Stats API keeps the abbreviation used that season.
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.teams (
  CONSTRAINT team_key_present EXPECT (team_id IS NOT NULL AND season IS NOT NULL)
    ON VIOLATION FAIL UPDATE
)
COMMENT 'MLB teams by season. One row per team and season.'
AS
SELECT
  t.value:id::INT AS team_id,
  t.value:season::INT AS season,
  t.value:name::STRING AS team_name,
  t.value:teamName::STRING AS short_name,
  t.value:abbreviation::STRING AS abbreviation,
  t.value:locationName::STRING AS location,
  t.value:league.id::INT AS league_id,
  t.value:league.name::STRING AS league_name,
  t.value:division.id::INT AS division_id,
  t.value:division.name::STRING AS division_name,
  t.value:venue.id::INT AS venue_id,
  t.value:venue.name::STRING AS venue_name,
  s._source_file,
  s._source_file_modified_at
FROM statsapi_latest s,
  LATERAL variant_explode(s.payload:teams) AS t
WHERE s.endpoint = 'teams';
