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
| `pitches_quarantine` | pitch | Pitches that failed a data quality rule, with the original values |
| `pitches_daily_quality` | date | Played games vs. pitches loaded, rule results, and any issues |
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
  doesn't stop the pipeline. A missing key fails the update. Everything else goes through
  the data quality rules below.

## Data quality

Pitches pass through `pitch_checks` (a pipeline-internal view) on the way to silver. Each
rule is also a pipeline expectation, so every update records pass and fail counts per rule
in the event log.

| Rule | Kind | Fails when |
|---|---|---|
| `pitch_key_present` | key | Game, at-bat or pitch number is missing. **Fails the update.** |
| `players_present` | row | Pitcher or batter is missing. |
| `valid_count`, `valid_outs` | row | Balls not 0–3, strikes not 0–2, or outs not 0–2. |
| `known_game` | row | No played game in `games` with this `game_pk` on this date. |
| `unique_pitch_key` | row | The same pitch key appears more than once. |
| `complete_plate_appearance` | row | The plate appearance's pitch numbers aren't 1..n (a pitch is missing). |
| `plausible_velocity` | value | Release or effective speed outside 25–110 mph. |
| `plausible_spin` | value | Spin outside 0–4000 rpm, or under 300 rpm on anything but a knuckleball, eephus or position player's lob. Spin axis outside 0–360°. |
| `plausible_location` | value | More than 10 ft to either side of the plate, or below −10 ft or above 15 ft. |
| `plausible_release` | value | Release more than 6 ft to the side, below 0.3 ft or above 9 ft, or extension outside 2.5–9.5 ft. |
| `plausible_strike_zone` | value | Zone bottom outside 0.5–3 ft, top outside 1.5–5 ft, or bottom above top. |
| `known_pitch_type` | value | A pitch type code that isn't in the known list. |

**What happens to a failing pitch**
- **Row rule:** the pitch can't be trusted, so it's left out of `pitches` (`action = 'rejected'`).
- **Value rule:** the pitch was thrown and its count and outcome are fine, so it stays. The
  failing group of measurements is set to NULL, the same as an ordinary tracking gap
  (`action = 'values_nulled'`). For example, a bad location nulls `plate_x`, `plate_z` and
  `zone`.
- Either way, the pitch lands in `pitches_quarantine` with `failed_rules` and the
  `original_values` the rules checked.

**The ranges are wide on purpose.** They catch readings that can't be real, not unusual
pitches. These all pass: position-player lobs (30–40 mph, about 3 ft of extension), Tyler
Rogers' submarine release (under 1 ft high), knuckleballs with almost no spin, and spiked
pitches (Statcast extrapolates the trajectory through the bounce, so `plate_z` goes negative).

**Results on 2023–2026.**
- Every row rule passes on all 2,912,445 pitches.
- 125 pitches have values nulled. 118 are spin misreads, such as 94 mph sinkers at about
  80 rpm, and 93 of those are from 2026. The rest are 4 locations (one pitch reads 57 ft
  underground), 3 release points and 1 velocity.

**Daily check.** `pitches_daily_quality` has one row per date. It lists `issues`:
- `games_missing_pitches`: a played game with no pitches
- `game_with_few_pitches`: a game with fewer than 100 pitches. The fewest in a real game
  was 143, in a game called after five innings, so fewer usually means a partial file.
- `pitches_rejected`: any rejected pitch

The check is per game rather than comparing a day's total with other days. The number of
games per day varies a lot, but pitches per game hardly does.

**Alerting.** The `ingest` job ends with a `check_data_quality` task. It reads the daily
check for the job's date window, prints a summary and the quarantined pitches by rule, and
fails the run if any day has issues. A failed run emails the deploying user.
- A missing game on the newest day is only a warning, because Savant sometimes posts a day
  late. The next run re-fetches that day, and if the game is still missing, it fails.
- The data is published either way; the check only raises the alarm.

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

## Feature tables

Model features live in the `ml` schema as Feature Engineering tables (`batter_profile`,
`pitcher_arsenal_features`), built from `fact_pitch` by the
`features` job. See [features.md](features.md).

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

## Statcast quirks

- **Tracking system.** Hawk-Eye has tracked every MLB park since 2020. Before that, pitches were
  tracked by PITCHf/x (through 2016) and TrackMan (2017–2019). All of our data (2023 on) comes
  from Hawk-Eye. Extending it to earlier seasons would mix tracking systems whose spin and
  release measurements aren't directly comparable.
- **Tracking gaps.**
  - About 0.4% of pitches have no velocity or location, and 0.5–1% no spin rate.
  - About 1,100 pitches over four seasons were not tracked at all: no pitch type, no speed.
    They still count, since the count and outcome are recorded.
  - Arm angle is missing on 0.5–1.3% of pitches, and about 0.3–0.4% of balls in play have no
    exit velocity.
- **Pitch clock.** Violations are rows with `description` `automatic_ball` or
  `automatic_strike` and no pitch type, since no pitch was thrown.
- **Bat tracking** (`bat_speed`, `swing_length`, ...) starts on 2023-07-14. It covers about
  43% of 2023 swings and 94–98% of swings from 2024 on.
- **Pitch-type reclassification.**
  - MLB assigns pitch types and can revise them after the game. Savant's overnight
    corrections are picked up by re-fetching the last few days on every run. Later changes
    to older days only arrive with a `--refetch-all` backfill.
  - The categories themselves changed in 2023, when sweeper (`ST`) and slurve (`SV`) were
    split out of slider (`SL`). From 2023 to 2026 sweepers rose from 6.5% to 8.3% of
    pitches and sliders fell from 15.5% to 13.0%. That's partly pitchers changing and
    partly pitches being labelled differently.
  - Pitch-type labels aren't a stable identity across seasons. Models should lean on pitch
    shape (velocity, movement, spin) or broad pitch groups.

## Known gaps

- A game stopped mid-inning and never resumed gives that half-inning a final state worth 0.
- `players` attributes come from the latest season a player appears in; the player lists
  don't record which team a player was on in each season.
