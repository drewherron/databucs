# Design: how databucs works and why

This document explains the design as a whole: what the system does, the main decisions
and the alternatives behind them, how the results were checked, and where the limits are.
The other docs cover each part in detail:
- [data_model.md](data_model.md): tables, data quality
- [features.md](features.md): feature tables
- [outcome_models.md](outcome_models.md): models, run values
- [recommender.md](recommender.md): batch recommender, dashboard, backtest
- [serving.md](serving.md): real-time endpoint
- [requirements_pitching_coach.md](requirements_pitching_coach.md): what the report is for

## What it does

Before a series, a pitching coach and an advance scout want to know two things: how each
of our pitchers should attack each hitter, and which pitchers match up best. databucs
answers that from public Statcast data. For every Pirates pitcher against every opposing
hitter, in a set of game situations, it scores each pitch the pitcher throws aimed at each
part of the zone by **expected run value**. It then groups the options into tiers (strong,
fine, avoid) relative to what the pitcher usually does.

It's prep before a game, not pitch calling during one: a plan per hitter, plus a "what if"
endpoint for situations the plan doesn't cover.

## Architecture

```
MLB Stats API, Baseball Savant
  │  fetch_raw (wheel task): raw files, content-addressed, into a UC volume
  ▼
bronze ── Auto Loader streaming tables (strings / VARIANT, as landed)
  ▼
silver ── typed entities + data quality rules + quarantine           ┐ lakehouse
  ▼                                                                   │ pipeline (SQL)
gold ──── fact_pitch (game state, outcomes, run value), run expectancy┘
  ▼
ml ────── feature tables (batter profile, pitcher arsenal), point in time
  ▼
five LightGBM outcome models ── MLflow, UC registry, @champion / @challenger
  ▼
recommender ─┬─ batch: pitch_option_scores ── AI/BI dashboard "Series prep"
             └─ real time: pyfunc + feature snapshot ── Model Serving endpoint
```

Everything is one Databricks Asset Bundle (dev and prod targets) with one Python wheel, on
Free Edition serverless compute.

| Databricks feature | Used for |
|---|---|
| Unity Catalog (catalog, schemas, volume) | all tables, raw files, models |
| Lakeflow Declarative Pipelines (SQL) | bronze → silver → gold, expectations, quarantine |
| Auto Loader | landing files into bronze, each exactly once |
| Lakeflow Jobs (wheel tasks, for_each, run_job) | ingest, features, training, recommender, serving |
| Feature Engineering in UC | point-in-time feature tables, lineage to models |
| MLflow + UC model registry | experiments, metrics, aliases (@champion / @challenger) |
| Model Serving | the real-time recommender endpoint |
| AI/BI dashboards | the coach-facing report, deployed by the bundle |

## Design decisions

### Data

- **Only 2023 onward.** The 2023 rule changes (pitch clock, shift ban, bigger bases)
  changed the game. Every season since 2020 is tracked by Hawk-Eye, so the data is one
  consistent tracking system.
  - *Alternative:* more seasons with an era flag. More rows, but different rules and, before
    2020, different tracking systems.
- **Raw files first, content-addressed.** Each response lands as an immutable file named by
  its hash. Re-running a window writes nothing new, a corrected re-pull lands next to the
  old file, and silver keeps the latest. Bronze is then a faithful, replayable record.
  - *Alternative:* fetch straight into tables. Simpler, but you can't replay or audit a
    load.
- **Bronze as strings and VARIANT; casting in silver.** Lenient casts (`try_cast`) mean
  one bad value doesn't stop the pipeline, while a missing key fails the update.
- **Data quality: reject, or null the bad values.** A pitch whose count, players or key is
  wrong is rejected. A pitch with an impossible measurement keeps its count and outcome,
  with only that group of measurements set to NULL. Both kinds go to a quarantine table
  with the original values, and a daily per-game check fails the ingest job on gaps.
  - *Alternative:* dropping whole rows for any problem. That loses real plate appearances
    over one misread spin rate.
- **Join on IDs, never team abbreviations** (Statcast relabels past seasons). Use
  `official_date` as a game's date (resumed games).

### Run value: the objective

- **Optimize expected run value, not strikes.** "Most likely strike" recommends pitches
  down the middle, which get strikes and get hit hard. Every option is valued by its
  expected change in run expectancy given the count, outs and runners.
- **Our own run values, with the base-out state.** Savant's `delta_run_exp` mostly ignores
  runners: a grand slam is worth less than a solo homer. The recommender scores options
  in a full situation, so it uses run expectancy with base-out state and count (RE24 plus
  count). It correlates about 0.84 with Savant's.
- **Values from the training seasons only.** The test season can't leak in through the
  run values.
- **Pooled where the effect is the same.** A called strike, a whiff and a foul with fewer
  than two strikes all just add a strike, so they share one value. Plate-appearance-ending
  results are valued by (outs, runners, result), not by count. An earlier version shrank
  sparse states toward the count average, and that wrongly pulled a bases-loaded home run
  from +3.2 to +2.0 runs.

### Features

- **Feature tables in Unity Catalog, written by a job.** Batter profiles and pitcher
  arsenals are written with the Feature Engineering client, with primary keys and a time
  key, so models get point-in-time lookups and lineage.
  - *Alternative:* pipeline materialized views with PK constraints, which needed the
    preview channel.
- **Point in time by construction.** A row's `as_of_date` is the day after the last game
  day it includes, and lookups take the newest row on or before the game's date. A game's
  own pitches never leak into its features. That includes the second game of a
  doubleheader, which can't see the first. A check task verifies 0 leaks after every build.
- **Trailing 365 days, shrunk toward the league.** A batter with little data looks
  average, and one with a lot looks like himself. The prior sizes are judgment calls.
- **Batter "cells"** (pitch group × in or out of zone × count group): the batter's swing,
  whiff and contact quality where *this* pitch would be.
- **Pitch shape over pitch labels.** Labels change across seasons (sweeper split out of
  slider in 2023), so the models get velocity, movement, release, and differences from the
  pitcher's primary fastball.
- **Game state is a request-time input, not a lookup.** It describes the situation being
  scored. A first version looked it up by pitch key, which made hypothetical pitches
  impossible to score, so it was retired.

### Outcome models

- **Five models along an outcome tree, not one.** swing → take outcome / whiff → contact
  outcome → in-play outcome. Each step is checked and calibrated on its own, and
  multiplying along the tree gives the probability of each of ten results.
  - *Alternative:* one multiclass model over all results. Fewer models, but much harder to
    diagnose.
- **LightGBM with fixed hyperparameters.** It's fast on millions of rows, handles missing
  values, and is well calibrated here without a calibration step (ECE ≤ 0.013).
  - *Alternative:* AutoML, which isn't available on Free Edition.
- **Train 2023–2025, test 2026, early stopping on held-out games.** The splits are by game,
  so a game's pitches never sit on both sides.
- **Fit in Free Edition memory.** It loads compact float32 columns, frees frames as soon as
  they're binned, and fits on at most about 1.5M rows, sampled by whole games.
- **Champion/challenger in the UC registry.** A new version takes `@champion` only if it
  beats the current champion's test log loss. It's forced when the inputs change.
  Retraining with the same settings moves log loss by about 0.001, so that's the noise
  floor for promotion.

### Recommender

- **Options are the pitcher's real arsenal × target zones.** Pitch types with ≥3% usage,
  each with his average velocity and movement, aimed at Statcast zones 1–9 and the four
  chase quadrants. That keeps the advice to things he actually throws.
- **Aim and miss.** A pitcher aims somewhere, and the pitch lands around it.
  - The models score a grid of 221 landing spots per pitch type.
  - Each aim point is the average over where it lands, with a normal miss of 0.65 ft.
  - *Why:* without misses, corners looked best everywhere ("paint the corner at 3-1"), and
    every option looked 0.04–0.10 runs per pitch better than reality.
  - With misses, the advice follows the count: behind → aim at the middle; 0-2 → expand.
- **Compared with the pitcher's usual mix, valued the same way.** The usual mix is valued
  at where his pitches really landed. So both sides of "vs usual" include misses.
  - It uses the exact count, falling back to the count group for thin samples. Pooling a
    whole group made full count look chase-heavy.
  - It answers the coach's question "is this a change from what he does?"
- **Tiers, not ranks.** The best and second-best options are often 0.001–0.004 runs apart,
  inside their own ranges. So options are *strong* (≥0.5 runs per 100 pitches better than
  his usual), *avoid* (≥1 worse) or *fine*. The thresholds are judgment calls.
- **Plain language.** Pitch names, not codes. Locations from the hitter's point of view
  ("chase low and in"), not the catcher's view or zone numbers.
- **No look-ahead.** Every input comes from before the series starts, so a demo on a past
  series sees what a real run that morning would have.
- **Series, lineup and staff.** It targets the next series within 10 days, or a demo on the
  last one played. Lineups are projected from recent starts, since teams post them hours
  before a game, and a custom lineup can be passed. Pitchers are the recent staff plus the
  probable starters.

### Report

- **Requirements first.** The dashboard is built against a one-page brief written as if
  from a pitching coach: who uses it, the questions it must answer, and what makes it
  usable.
- **AI/BI dashboard as a bundle resource.** It's versioned with the code, and its datasets
  point at the target's schema automatically.
- **The output table is self-contained.** Dashboard queries can only use one schema, so
  the recommender writes labels, tiers and names into `pitch_option_scores` itself. A
  later printed report will read the same definitions.

### Real-time serving

- **A feature snapshot inside the model.** A job pulls a dated snapshot of everything the
  recommender looks up, plus the champion boosters, into one pyfunc. It registers the
  pyfunc in UC and swaps the endpoint to the new version.
  - *Alternative:* an online feature store. Online tables aren't available on Free
    Edition, and lookups run on request rows *before* the model, while the recommender
    expands one request into every pitch type × landing spot *inside* it.
- **Two engines, held together by a check.** The batch job is Spark SQL and the endpoint is
  pandas. They share the geometry, tiers, labels and input queries, and every registration
  recomputes 20 batch situations and fails if they differ. The largest difference so far
  is 3e-17 runs.
- **CPU, small, scale to zero.** It's cheap and allowed on Free Edition. The cost is a cold
  start after idling.

### Engineering

- **One bundle, dev and prod in one workspace.** Dev mode prefixes schemas and jobs per
  user, so development never touches prod tables.
- **Serverless only.** Free Edition can't resize compute. That shaped the memory work and
  ruled out `cache()` (not supported on serverless).
- **A plain Python package, tested locally.** Job logic lives in `src/databucs`, with unit
  tests that need no cluster (SQL builders, geometry, transforms, the pandas recommender
  with stub models). CI runs ruff and pytest.

## How it was checked

| Check | Result |
|---|---|
| Data quality rules | all row rules pass on 2,912,445 pitches; 125 had values nulled |
| Point-in-time features | 0 leaks; window counts match a direct recount |
| Outcome models, 2026 test season | swing AUC 0.881, whiff 0.796; log loss well below baselines; ECE ≤ 0.013 |
| Expected run value, 2026 | deciles within 0.004 runs; correlation 0.22 with a pitch's actual run value (count baseline 0.01) |
| Within-situation backtest, 2026 | slope 1.05 ± 0.01 (0.96 ± 0.02 within in or out of the zone); best third of pitches gave up 7.0 fewer runs per 100 than the worst, 6.4 predicted |
| Recommender vs real counts | model averages by count and zone match 2023–2025 results (e.g. 3-1 in zone −0.063 vs −0.059) |
| Batch vs real-time engine | identical to 3e-17 runs on 20 sampled situations |

## Strengths

- **The objective is the right one:** runs, with the full base-out state and count, not
  strikes or whiffs.
- **Honest about command.** It's one of the few public-data approaches that values *aiming*
  at a spot rather than a perfect hit of it, and the miss model visibly changed the advice
  toward sound baseball.
- **Leak-free by construction,** from content-addressed raw files to point-in-time
  features to a held-out test season and run values from training seasons only.
- **Checked at the level decisions are made.** The backtest compares pitches within the
  same situation, which is the comparison the recommender actually makes.
- **Built for the people who use it:** requirements first, tiers instead of false
  precision, plain-language locations, "vs his usual" as context.
- **Production habits on a free workspace:** one bundle, dev/prod separation, data quality
  alerts, champion/challenger, a consistency check between engines, unit tests and CI.

## Limitations

- **Selection bias.** Every model learns from pitches pitchers *chose* to throw, for
  reasons the data can't see (the scouting report, the previous pitch, the catcher's
  call). An option pitchers rarely throw in a situation is less tested than its numbers
  suggest. Modeling pitch shape and location rather than labels reduces this, but doesn't
  remove it.
- **One miss size for everyone.** Command varies by pitcher and pitch type (breaking balls
  miss more), but there's one league-wide 0.65 ft spread.
  - For pitchers with better command, the usual mix can beat every aimed option, and
    everything shows as "avoid". That's 0.5–7% of demo situations, concentrated in a few
    pitchers.
  - A per-pitcher estimate needs intended targets, which public data doesn't have.
- **No sequencing or predictability.** Each pitch is scored alone. The previous pitch,
  tunneling, and how predictable a choice is are ignored.
- **What the backtest doesn't cover.** It scores pitches where they landed, so the miss
  model and the tiers themselves aren't directly validated. A full-tier backtest would need
  about 23M scoring rows for the Pirates alone.
- **What the models can't see.**
  - Contact quality is mostly decided at contact, so the contact and in-play models
    improve only modestly on baselines.
  - Hit-by-pitch is under-predicted (0.21% vs 0.30% in 2026).
  - Wild pitches and passed balls aren't modeled, so a splitter in the dirt with a runner
    on third looks safer than it is.
- **Fixed situations.** Inning, score and times through the order are fixed per scenario.
  Late-and-close needs win expectancy, which isn't built yet.
- **Thin samples.** Shrinkage handles a batter with little data by pulling him toward the
  league, not toward similar hitters. Rookies and new pitch types get generic advice.
  - A pitcher with little data (under 500 pitches) is flagged "limited data" rather than
    hidden: his pitch measurements settle within a few dozen pitches, and his results
    features are shrunk toward the league.
  - A rarely thrown pitch is a limit no model fixes: its results are mostly luck, and
    pitchers may avoid it for reasons the data can't see. Only established pitches (100
    thrown, or 10% of a newer pitcher's mix) are recommended, so a pitch he's still
    developing never shows up.
- **Judgment calls.** Shrinkage priors, tier thresholds, the miss size and the scenario
  defaults are reasoned choices, not fitted. They're documented where they live.
- **Freshness.** The endpoint answers from a dated snapshot until the job reruns. Prod
  isn't deployed or scheduled yet, and the dashboard's filter defaults point at the demo
  hitter and pitcher.

## Future additions

Roughly in order of value:

1. **Better matchup rankings.** The [simulator](simulator.md) and the dashboard's
   Matchup rankings page give runs, K% and BB% per pitcher × hitter. Next: widen the
   compressed spread between pitchers (a pitcher's own past results), reliever tiers for
   each three-batter part of the order, and starters by time through the order.
2. **Production deploy and scheduling.** Deploy `prod`, backfill, and schedule ingest →
   features → recommender before each series, with `register_recommender` daily in season.
3. **Similar hitters** (Vector Search, which allows one endpoint on Free Edition). Back up
   advice for thin-sample batters with comparable hitters, and show them in the report.
4. **Next-pitch model and predictability.** Flag predictable patterns in a pitcher's
   sequencing, and stop recommending options that are too predictable.
5. **Better command modeling.** A miss size per pitch group, or fitted per pitcher from
   location scatter with care: scatter mixes intended variety with misses.
6. **Win expectancy** for late-and-close situations, and wild-pitch risk with runners on.
7. **Monitoring.** Drift and calibration over time. Try Lakehouse Monitoring by creating a
   monitor on one table (availability on Free Edition is unclear), or fall back to a
   scheduled drift job in SQL.
8. **Report extras.** A printable per-hitter report, and a short plain-English summary per
   hitter with `ai_query` that may only state numbers from the tables.
9. **Smaller refinements.**
   - empirical-Bayes shrinkage priors
   - a margin or several seeds for champion promotion
   - per-pitcher scenario defaults (a starter's real times through the order)
   - an online feature store as a learning exercise
