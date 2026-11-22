# databucs

An end-to-end data and ML platform on Databricks that produces advance-scouting reports: per-batter pitch recommendations and Pirates pitcher matchup rankings, built from public Statcast data.

> Personal project to learn Databricks. Not affiliated with the Pittsburgh Pirates or MLB.

## Status
- [x] Project setup: Asset Bundle (dev/prod), Unity Catalog schemas, CI
- [x] Statcast and MLB Stats API ingestion into bronze
- [ ] Silver/gold data model

## Layout
```
databricks.yml        # Asset Bundle config (dev/prod targets, wheel build)
resources/            # schemas, volume, jobs, pipeline
src/databucs/
  ingestion/          # Savant + MLB Stats API clients, landing-file writer
  jobs/               # job entry points
src/pipelines/        # Lakeflow declarative pipeline (SQL)
tests/
.github/workflows/    # CI: lint + unit tests
```

## Setup
Requirements: [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install),
[uv](https://docs.astral.sh/uv/), a Databricks workspace (built on Free Edition).

```bash
uv sync                                   # local env (Python 3.12, matches serverless)
uv run pytest                             # unit tests
databricks auth login --host <workspace-url>
databricks bundle validate                # check config
databricks bundle deploy                  # deploy to dev (default target)
databricks bundle run smoke_test          # verify compute + schemas
```

Dev and prod live in the same workspace. Development mode prefixes schemas per user (`dev_<user>_bronze`), while prod uses plain names (`bronze`), all in the `workspace` catalog. Deploy prod with `databricks bundle deploy -t prod`.

## Ingestion
The `ingest` job runs two tasks:

1. **`fetch_raw`** (Python wheel) downloads raw data and writes it unchanged to the
   `bronze.landing` volume:
   - MLB Stats API schedule, one file per month, with probable pitchers and starting lineups
   - Statcast pitch-level CSV from Baseball Savant, one file per day with a completed game
     (regular season and postseason)
   - MLB Stats API teams and players, one file per season
2. **`run_lakehouse`** runs the `lakehouse` pipeline, where Auto Loader picks up new files into
   `bronze.statcast_pitches` (all columns as strings) and `bronze.statsapi_responses`
   (whole responses as `VARIANT`), and the silver tables refresh from bronze.

Landing files are content-addressed (`statcast_2025-07-04_<sha256 prefix>.csv`), so re-running
a window writes nothing new, and a corrected re-pull lands as a new file next to the old one.
Days older than the lookback window that already have a file are skipped, so an interrupted
backfill resumes where it stopped.

```bash
databricks bundle run ingest                                  # incremental: last 3 days
databricks bundle run ingest --params start_date=2025-03-01,end_date=2025-11-30   # backfill
uv run fetch-raw --landing-path ./landing --start-date 2025-07-01 --end-date 2025-07-03  # local
```

## Silver tables
- **`silver.pitches`**: one row per pitch, keyed on `(game_pk, at_bat_number, pitch_number)`,
  with typed columns and clearer names (`pitcher_id`, `outs`, `times_through_order`, ...).
  When a day has been re-pulled, only the latest file for that date is kept. Rows with a
  missing key fail the update; invalid counts, outs, or missing players are flagged by
  pipeline expectations.
- **`silver.games`**: one row per game from the schedule, with status, teams, score, venue,
  and the announced probable starters. Postponed and resumed games appear in the schedule
  more than once; the played appearance is kept. `official_date` is the date Statcast files
  the game's pitches under, so it is the date to join on.
- **`silver.lineups`**: starting lineups, one row per game, team, and batting-order spot.
- **`silver.teams`**: one row per team and season (abbreviations and divisions can change).
  Join pitches to teams through `games` team IDs, not abbreviations: Statcast relabels past
  seasons with a team's current abbreviation (`ATH` for 2023), the Stats API doesn't (`OAK`).
- **`silver.players`**: one row per player, with handedness, size, and debut date from the
  latest season the player appears in.
