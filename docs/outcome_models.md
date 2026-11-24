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
- **The game state:** balls, strikes, outs, inning, score difference, runners and times
  through the order. `transform()` derives the rest from these: count group, runners on,
  runner in scoring position, late and close, first pitch of the plate appearance, and
  same-side matchup.
- **Lookups from the feature tables** (see [features.md](features.md)):
  - the batter's profile, plus the batter's rates in this pitch's own cell (pitch group ×
    zone or chase × count group)
  - the pitcher's arsenal for this pitch type

**Request-time inputs vs lookups.** The pitch and the game state describe what is being
scored, so every row supplies them. Only history (batter, arsenal) is looked up, by player,
pitch type and date. So a hypothetical pitch, such as "slider low and away at 1-2 with a
runner on third", can be scored as easily as a real one: it needs no pitch key. The
logged models' inputs are the lookup keys (`game_date`, `batter_id`, `pitcher_id`), the
pitch columns and the state columns. `game_pk` is in the training rows only to split them
by game, and `exclude_columns` keeps it out of the logged inputs.

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
  200 rows per leaf, and up to 8,000 rounds with early stopping. The swing model was still
  improving after 3,000 rounds. A learning rate of 0.1 made it stop at 1,700 but tested
  worse, so it gets more rounds instead.
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
registered in Unity Catalog as `<ml schema>.outcome_<name>`.

**Champion and challenger.** A new version becomes `@champion` only if its test log loss
beats the current champion's on the same test season. Otherwise it gets `@challenger`, to
compare by hand. Run with `--params promote=always` when the models' inputs change, since
an old champion then can't stand in for the new one.

**Serverless retries**

Serverless jobs retry failed tasks on their own, including after running out of memory,
even with `max_retries: 0`. Each successful attempt registers a version.

## Results on 2026 (test season)

| Model | Log loss | Best baseline | AUC | Best baseline AUC | ECE (worst class) |
|---|---|---|---|---|---|
| swing | 0.430 | 0.659 (count) | 0.881 | 0.640 | 0.013 |
| take_outcome | 0.132 | 0.601 (count) | | | 0.006 |
| whiff | 0.443 | 0.552 (pitch type) | 0.796 | 0.602 | 0.013 |
| contact_outcome | 0.659 | 0.688 (pitch type) | 0.642 | 0.553 | 0.004 |
| in_play_outcome | 0.905 | 0.925 (pitch type) | | | 0.008 |

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
  looks up the batter and arsenal features itself.
- Expected run value = Σ P(result) × run value(result, state).
- The predictions are in `<ml schema>.outcome_test_predictions`.

**Results on 722,008 pitches in 2026**
- **Result shares:** predicted and actual shares of the ten results agree closely.
  - ball 36.5% vs 36.0%, whiff 12.2% vs 12.0%, home run 0.76% vs 0.78%
  - The biggest gap is hit by pitch, at 0.21% predicted vs 0.30% actual.
- **Ranking:** actual run value rises steadily with expected run value across deciles,
  from −0.090 in the lowest to +0.109 in the highest.
- **Calibration:** each decile's mean expected value is within 0.004 runs of its actual
  mean.
- **Error:**
  - Correlation with a pitch's actual run value is 0.22, against 0.01 for the count and
    base-out baseline.
  - Mean squared error falls by 5% compared with predicting zero.
  - A single pitch's run value is mostly noise, so 0.22 at the pitch level is a real
    signal.

## Known limitations and next steps

- **Run-to-run noise in promotion:** retraining with the same settings moves test log loss
  by about 0.001. The swing model scored 0.4291 when it ran out of rounds at 3,000, and
  0.4299 when it stopped early at 5,734. `promote=if_better` treats any gain as real, so a
  margin, or a comparison over several seeds, would make promotion more robust.
- **Selection bias:** the training data only has pitches that were actually thrown, so the
  models learn how pitches perform when pitchers chose to throw them. See the notes
  on selection bias in [design.md](design.md#limitations).
- **Hit-by-pitch rate:** HBPs are under-predicted for 2026 (0.21% vs 0.30%). The 2026 rate
  may simply be higher than in the training seasons.
