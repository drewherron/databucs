# Data model

All tables are built by the `lakehouse` pipeline (Lakeflow Declarative Pipelines, SQL) in the
`workspace` catalog. Development deployments prefix schemas per user (`dev_<user>_silver`).

```
landing volume ──► bronze ──► silver ──► gold
(raw files)        (as landed) (typed, one row per entity) (query- and model-ready)
```

## Bronze: raw, append-only

| Table | Grain | Notes |
|---|---|---|
| `statcast_pitches` | pitch × landed file | Savant CSV columns as strings. A re-pulled day adds a second copy. |
| `statsapi_responses` | landed file | Whole Stats API response as `VARIANT`, plus `endpoint`. |

Streaming tables fed by Auto Loader, so each landed file is read exactly once. Every row
keeps `_source_file`, `_source_file_modified_at` and `_ingested_at` for lineage.

## Silver: cleaned entities

| Table | Grain (key) | Source |
|---|---|---|
| `pitches` | pitch (`game_pk`, `at_bat_number`, `pitch_number`) | Statcast |
| `games` | game (`game_pk`) | Stats API schedule |
| `lineups` | starting-lineup spot (`game_pk`, `team_id`, `batting_order`) | Stats API schedule |
| `teams` | team-season (`team_id`, `season`) | Stats API teams |
| `players` | player (`player_id`) | Stats API players |

Materialized views, refreshed from bronze on every pipeline run (incrementally where the
engine can).

**Design decisions**

- **Latest version wins.** A corrected re-pull lands as a new file. `pitches` keeps only the
  latest file per game date, so pitches removed by a correction disappear too. The Stats API
  tables keep the latest file per request (`statsapi_latest`).
- **One schedule entry per game.** Postponed and resumed games are listed on more than one
  date. `schedule_games` keeps the played entry, then the latest date.
- **`official_date` is the game's date.** Statcast files all pitches of a resumed game under
  the day it started, which is the Stats API's `officialDate`.
- **Join on IDs, not abbreviations.** Statcast relabels past seasons with a team's current
  abbreviation (`ATH` in 2023), while the Stats API keeps the one used that season (`OAK`).
  Pitches get their team IDs through `games`.
- **Lenient casts, strict keys.** Values are cast with `try_cast`, so one malformed value
  doesn't stop the pipeline. A missing key fails the update. Out-of-range counts, outs, and
  missing players are recorded as expectation violations.

## Gold: analysis and model inputs

| Table | Grain (key) | Purpose |
|---|---|---|
| `fact_pitch` | pitch (`game_pk`, `at_bat_number`, `pitch_number`) | One wide row per pitch: names, teams, game state, pitch traits, outcome flags, run value. |
| `ref_run_expectancy` | `season`, `outs`, `base_state`, `balls`, `strikes` | Average runs to the end of the half-inning from each state. |

`fact_pitch` is clustered on `pitcher_id`, `batter_id`, `game_date`, the usual filters.

**Game state.** Statcast records the state *before* each pitch. The state *after* a pitch is
the state before the next pitch in the same half-inning (the `pitch_states` view).
`base_state` is a three-character code: `1_3` means runners on first and third.

**Run expectancy.**
- `ref_run_expectancy` is computed from our own data: regular-season innings 1–8, by season.
  Walk-offs end the late innings early, and extra innings start with a runner on second.
- The 0-0 rows are the standard 24-state (RE24) table, since every plate appearance starts
  at 0-0. The other counts give the value of the count itself.
- `sample_size` is kept because rare states, like bases loaded at 3-0, have small samples.

**Run value.**
- `run_value = runs scored on the pitch + RE(state after) − RE(state before)`, from the
  batting team's side. Positive is good for the hitter.
- The state after the last pitch of a half-inning is worth 0.
- Walk-off pitches have a NULL `run_value`. The game ends before the third out, so the state
  after has no meaningful run expectancy.

**`run_value` vs Savant's `delta_run_exp`.** Both columns are kept (`savant_run_value`), but
they measure different things.
- Savant's value largely ignores the runners on base. A grand slam is worth about the same as
  a solo home run (≈1 run), and every double play is worth about −0.3.
- `run_value` uses the full base-out state: a double play with runners on first and second
  and nobody out is worth about −1.2, and a grand slam about +2.
- They agree where runners don't matter, such as bases-empty balls and strikes, and correlate
  at about 0.84 overall.
- The recommender scores options in the actual situation, so it uses `run_value`.

## Example

Every slider a pitcher threw to left-handed hitters with two strikes, with its run value:

```sql
SELECT game_date, batter_name, count_state, base_state, outs, description, run_value
FROM gold.fact_pitch
WHERE pitcher_name = 'Paul Skenes'
  AND pitch_name = 'Slider'
  AND batter_stands = 'L'
  AND strikes = 2
ORDER BY game_date;
```

## Known gaps

- About 0.4% of pitches have no velocity or location, and 0.7% no spin rate (tracking gaps).
- A game stopped mid-inning and never resumed gives that half-inning a final state worth 0.
- `players` attributes come from the latest season a player appears in; the player lists
  don't record which team a player was on in each season.
