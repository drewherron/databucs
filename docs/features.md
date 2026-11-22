# Feature tables

Feature tables for the models, in the `ml` schema (`dev_<user>_ml` in development). They
are Unity Catalog tables with primary keys, written with Feature Engineering's
`FeatureEngineeringClient`, so models can join them with `FeatureLookup` and point-in-time
lookups.

| Table | Key | Time key | One row per |
|---|---|---|---|
| `batter_profile` | `batter_id` | `as_of_date` | batter, from the day after each day they batted |
| `pitcher_arsenal_features` | `pitcher_id`, `pitch_type` | `as_of_date` | pitcher × pitch type, from the day after each day they pitched |
| `game_state_features` | `game_pk`, `at_bat_number`, `pitch_number` | none | pitch |

All three are built from `gold.fact_pitch`. The SQL is in `src/databucs/features/`.

## Point-in-time correctness

`as_of_date` is the first day a row may be used, which is **the day after the last game
day it includes** (`batter_data_through`, `arsenal_data_through`).

- A point-in-time lookup for a pitch on day D takes the newest row with
  `as_of_date <= D`. That row only covers game days before D, so a game's own pitches never
  leak into its features. This includes the second game of a doubleheader, which can't see
  the first.
- For a future game, the same lookup returns each player's latest row.
- A player with no earlier pitches, such as a debut, gets NULL features. The same goes for a
  pitch type a pitcher hasn't thrown before. A model should
  treat NULL as "no information" or fill it with the league average.

Game state is known before the pitch is thrown, so `game_state_features` has no time key:
the row for a pitch is always safe to use for that pitch. To score a hypothetical
situation, such as a 3-1 count with runners on, the recommender passes these values in
directly instead of looking them up.

## Windows and shrinkage

- **Window:** profiles and arsenals cover the trailing 365 days of pitches, ending with the
  last game day included. On opening day that is the whole previous season.
- **Shrinkage:** batter rates are shrunk toward the league rate for the same cell over the
  same window:

  ```
  shrunk rate = (successes + k × league rate) / (trials + k)
  ```

  A batter with little data looks close to average, and one with a lot looks like
  themselves. The prior sizes `k` are judgment calls, not fitted:

  | Rate | Denominator | k |
  |---|---|---|
  | swing and chase rates | pitches | 40 |
  | whiff rate | swings | 30 |
  | xwOBA on contact | balls in play | 25 |
  | strikeout and walk rates | plate appearances | 60 |

  A fitted prior (empirical Bayes per rate) is a possible improvement.

## batter_profile

- **Overall rates:** pitches seen, plate appearances, zone swing rate, chase rate, whiff
  rate (per swing), xwOBA on contact, strikeout and walk rates, and handedness (`bats`).
- **Bat tracking:** average bat speed and swing length over swings of at least 50 mph, which
  leaves out bunts and most checked swings, with the number of tracked swings. These are
  not shrunk. Bat tracking starts in July 2023.
- **Cells:** 18 cells, from 3 pitch groups × 2 regions × 3 count groups. Each cell has its
  pitch count, swing rate, whiff rate and xwOBA on contact.
  - **Pitch groups:** `fastball` (FF, SI, FC), `breaking` (SL, ST, SV, CU, KC, CS) and
    `offspeed` (CH, FS, FO, SC). Eephus pitches, knuckleballs and position players' pitches
    count in the overall rates only.
  - **Regions:** `zone` (Statcast zones 1–9) and `chase` (outside them). In the chase
    region, the swing rate is the chase rate.
  - **Count groups:** `hitter_ahead` (more balls than strikes, under two strikes),
    `two_strikes`, and `neutral` (0-0, 0-1, 1-1).

For example, `breaking_chase_two_strikes_swing_rate` is how often the batter chases a
breaking ball with two strikes.

## pitcher_arsenal_features

Per pitch type, averaged over the window:
- usage rate
- velocity
- spin rate
- horizontal break and induced vertical break, in inches
- release side, release height, extension and arm angle

Horizontal values are flipped for right-handers, so positive always means arm side.

On each day a pitcher pitched, there's a row for every pitch type they threw in the window,
not just that day, so usage stays current for pitches they skipped.

**Primary fastball:** the most-thrown four-seamer, sinker or cutter in the window. Every
pitch type records its velocity, horizontal and vertical break differences from the
primary fastball, which is how hitters experience a secondary pitch. Pitch-type labels
change across seasons (see "Statcast quirks" in [data_model.md](data_model.md)), so models
should lean on these shape values rather than on the label.

## game_state_features

The situation before each pitch:
- count and count group
- outs, runners, runners on base, runner in scoring position
- inning, score difference for the batting team
- late and close (Elias definition)
- times through the order, first pitch of the plate appearance, and same-side matchup

## Building and checking

- **Updates:** the `features` job runs after every `ingest` run, for the same window. New
  rows are merged on the primary key, so re-running a window replaces just those rows. A
  table that doesn't exist yet is built from all of history.
- **Rebuilds:** after changing a feature definition, rebuild with
  `databricks bundle run features --params full_rebuild=true`.
- **Checks:** the job's `check_features` task builds a training set with
  `create_training_set` from the latest week of pitches and all three feature tables, then
  checks four things:
  - No row uses data from the pitch's game day or later.
  - Each batter row's `pitches_seen` matches a direct count over its window.
  - Every pitch has game state, and at least 90% have batter and arsenal features.
    Coverage is lowest in opening week, because of debuts and new pitch types. In 2026 it
    was 99.4% (batter) and 96.3% (arsenal) in opening week, and 99.8% and 98.6% in June.
  - A lookup for a game 30 days later returns each batter's latest row.

```python
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()
training_set = fe.create_training_set(
    df=labels,  # game_pk, at_bat_number, pitch_number, game_date, batter_id, pitcher_id, pitch_type, run_value
    feature_lookups=[
        FeatureLookup(
            table_name="workspace.ml.batter_profile",
            lookup_key="batter_id",
            timestamp_lookup_key="game_date",
        ),
        FeatureLookup(
            table_name="workspace.ml.pitcher_arsenal_features",
            lookup_key=["pitcher_id", "pitch_type"],
            timestamp_lookup_key="game_date",
        ),
        FeatureLookup(
            table_name="workspace.ml.game_state_features",
            lookup_key=["game_pk", "at_bat_number", "pitch_number"],
        ),
    ],
    label="run_value",
)
```

## Known limitations

- **Corrections:** a merge doesn't delete rows. If a Savant correction removes a player's
  appearance from a day, that day's feature row stays until a full rebuild.
- **Online lookup:** there is no online store yet. Free Edition doesn't support online
  tables, so real-time serving would pass features in rather than look them up.
