-- Starting lineups: one row per game, team, and batting-order spot (1-9). Comes from the
-- schedule entry for each game (see schedule_games), which lists the starting nine in
-- batting order. Games that weren't played have no lineup.
--
-- Position is the player's primary position as listed with the lineup, which is not always
-- the position played in that game.
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.lineups (
  CONSTRAINT lineup_key_present
    EXPECT (game_pk IS NOT NULL AND team_id IS NOT NULL AND batting_order IS NOT NULL)
    ON VIOLATION FAIL UPDATE,
  CONSTRAINT player_present EXPECT (player_id IS NOT NULL),
  CONSTRAINT valid_batting_order EXPECT (batting_order BETWEEN 1 AND 9)
)
CLUSTER BY (game_pk)
COMMENT 'Starting lineups by game and team. One row per batting-order spot.'
AS
WITH sides AS (
  SELECT game_pk, 'home' AS side, game:teams.home.team.id::INT AS team_id,
    game:lineups.homePlayers AS players
  FROM schedule_games
  UNION ALL
  SELECT game_pk, 'away' AS side, game:teams.away.team.id::INT AS team_id,
    game:lineups.awayPlayers AS players
  FROM schedule_games
)
SELECT
  s.game_pk,
  s.side,
  s.team_id,
  p.pos + 1 AS batting_order,
  p.value:id::INT AS player_id,
  p.value:primaryPosition.abbreviation::STRING AS primary_position
FROM sides s,
  LATERAL variant_explode(s.players) AS p;
