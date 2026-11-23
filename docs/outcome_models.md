# Pitch outcome models

Five LightGBM models predict what happens on a pitch, given the pitch (type, shape,
location), the batter's profile, the pitcher's arsenal and the game state. Combined with
run values, they give each pitch an **expected run value**, which is what the recommender
ranks options by.

## The outcome tree

A pitch's result is split into steps, each with its own model, so each step can be checked
and calibrated on its own:

| Model | Trained on | Predicts |
|---|---|---|
| `outcome_swing` | competitive pitches | take / swing |
| `outcome_take_outcome` | takes | ball / called strike / hit by pitch |
| `outcome_whiff` | swings | contact / whiff |
| `outcome_contact_outcome` | contact | foul / in play |
| `outcome_in_play_outcome` | balls in play | out / single / double / triple / home run |

Multiplying along the tree gives the probability of each of ten results. For example,
P(foul) = P(swing) × P(contact | swing) × P(foul | contact).

**Label decisions** (`src/databucs/models/outcomes.py`)
- **Competitive pitches only.** Training leaves out pitch-clock violations, pitchouts,
  catcher's interference and untracked pitches. It also leaves out bunt attempts, which are
  a tactical choice rather than a reaction to the pitch (`fact_pitch.is_bunt_attempt`).
- **Foul tips count as whiffs.** A foul tip is always a strike and an out with two strikes,
  which is a swinging strike's effect, not a foul's.
- **Errors and fielder's choices count as outs.** The model predicts batted-ball quality,
  not fielding. Run values still include what errors actually cost.

## Inputs

Each model has 113 input columns (`src/databucs/models/features.py`):
- **The pitch's own values:** velocity, spin, movement, release, and location relative to
  the batter. Location is horizontal distance toward the batter and height as a fraction of
  the batter's strike zone.
- **Lookups from the feature tables** (see [features.md](features.md)):
  - the batter's profile, plus the batter's rates in this pitch's own cell (pitch group ×
    zone or chase × count group)
  - the pitcher's arsenal for this pitch type
  - the game state

Point-in-time lookups through Feature Engineering keep every pitch's features to data from
before its game day. The pitch's speed and movement differences from the pitcher's primary
fastball come from the arsenal row.

One `transform()` function derives the final columns in both training and scoring, so the
two can't drift apart.

## Training (`train_outcome_models` job)

**Data and model**
- **Split:** train on 2023–2025 and test on 2026. 10% of training games (by `game_pk`) are
  held out for early stopping.
- **Model:** LightGBM with fixed hyperparameters: learning rate 0.05, 63 leaves, at least
  200 rows per leaf, and up to 3,000 rounds with early stopping.
- **Parallel runs:** the job is a `for_each_task` over the five models, run in parallel.

**Memory**

Free Edition compute is best-effort and can't be resized, so the job keeps memory down:
- It loads only the needed columns as float32.
- It frees each frame once it's binned.
- It fits on at most about 1.5M rows, sampled by whole games in Spark before `toPandas`.
  Only the swing model is above that.

**What MLflow records**
- parameters
- test metrics against baselines
- calibration tables and plots
- feature importance

**Registration**

Each model is logged with `fe.log_model`, so it carries its feature lookups. It's
registered in Unity Catalog as `<ml schema>.outcome_<name>` with the alias `@champion`.

**Serverless retries**

Serverless jobs retry failed tasks on their own, including after running out of memory,
even with `max_retries: 0`. Each successful attempt registers a version.

## Results on 2026 (test season)

| Model | Log loss | Best baseline | AUC | Best baseline AUC | ECE (worst class) |
|---|---|---|---|---|---|
| swing | 0.429 | 0.659 (count) | 0.881 | 0.640 | 0.009 |
| take_outcome | 0.132 | 0.601 (count) | | | 0.006 |
| whiff | 0.443 | 0.552 (pitch type) | 0.796 | 0.602 | 0.012 |
| contact_outcome | 0.659 | 0.688 (pitch type) | 0.642 | 0.553 | 0.005 |
| in_play_outcome | 0.904 | 0.925 (pitch type) | | | 0.006 |

The baselines are the class rates from the training seasons:
- **count:** by balls and strikes
- **pitch type:** by pitch type
- **prior:** overall

ECE (expected calibration error) is the row-weighted gap between predicted and observed
rates across 20 equal-size bins. The models are well calibrated without a separate
calibration step.

What the numbers say:
- **Take outcome:** location nearly decides ball versus called strike, so this model is
  very strong.
- **Swing and whiff:** these depend on location, count and the batter's habits, and the
  models beat the baselines by a wide margin.
- **Contact and in-play outcomes:** much of what happens after contact is decided at
  contact, which these models can't see. They improve modestly on the baselines, which is
  expected.

## Expected run value

`src/databucs/models/run_values.py` and the `evaluate_expected_run_value` task build the
run values from the training seasons only, so the test season can't leak in.

**Run values per result and state**
- The value is "runs scored + run expectancy after the pitch − run expectancy before",
  averaged over pitches.
- **Results that change the count** are averaged per full state, which is balls, strikes,
  outs and runners. A called strike, a whiff, and a foul with fewer than two strikes all
  just add a strike, so they share one value. That gives every state plenty of data, even
  3-0.
- **Results that end the plate appearance** don't depend on the count once they happen. A
  bases-loaded home run scores four whatever the count. Their value after the pitch is
  averaged per (outs, runners, result), and the count enters only through the run
  expectancy before the pitch.
  - Example: a home run with the bases loaded, two outs and a 3-2 count is worth +3.17.
  - Shrinking sparse states toward the count's average was tried first. It wrongly pulled
    that home run down to +2.0, because the state was rare, even though its value is
    almost fixed.

**Scoring 2026**
- Each `@champion` model scores the 2026 competitive pitches with `fe.score_batch`, which
  looks up the features itself.
- Expected run value = Σ P(result) × run value(result, state).
- The predictions are in `<ml schema>.outcome_test_predictions`.

**Results on 722,008 pitches in 2026**
- **Result shares:** predicted and actual shares of the ten results agree closely.
  - ball 36.6% vs 36.0%, whiff 12.1% vs 12.0%, home run 0.77% vs 0.78%
  - The biggest gap is hit by pitch, at 0.21% predicted vs 0.30% actual.
- **Ranking:** actual run value rises steadily with expected run value across deciles,
  from −0.090 in the lowest to +0.110 in the highest.
- **Calibration:** each decile's mean expected value is within 0.004 runs of its actual
  mean.
- **Error:**
  - Correlation with a pitch's actual run value is 0.23, against 0.01 for the count and
    base-out baseline.
  - Mean squared error falls by 5% compared with predicting zero.
  - A single pitch's run value is mostly noise, so 0.23 at the pitch level is a real
    signal.

## Known limitations and next steps

- **Swing model round limit:** the swing model used all 3,000 rounds without early
  stopping. A higher learning rate or more rounds should improve it a little.
- **The `season` input:** models registered before 2026-11-23 list `season` as an input,
  because it was in their training rows. Training now derives the season from `game_date`,
  so retrained models won't need it. The evaluation job passes it for now.
- **Champion promotion:** every training run moves `@champion` to the new version. Once
  there's something to compare, a challenger should only be promoted if it beats the
  champion on the test season.
- **Selection bias:** the training data only has pitches that were actually thrown, so the
  models learn how pitches perform when pitchers chose to throw them. See the plan's notes
  on selection bias; a write-up belongs in `docs/limitations.md`.
- **Hit-by-pitch rate:** HBPs are under-predicted for 2026 (0.21% vs 0.30%). The 2026 rate
  may simply be higher than in the training seasons.
