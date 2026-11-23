# Pitch recommender (batch)

For an upcoming series, the `recommend_pitches` job scores every Pirates pitcher's pitch
options against each opposing batter in a set of game situations, and writes them to
`<ml schema>.pitch_option_scores`. An option is a pitch type the pitcher really throws,
aimed at a zone. Its score is the **expected run value** from the
[outcome models](outcome_models.md), where lower is better for the pitcher.

```
databricks bundle run recommend_pitches                                  # next series, or a demo
databricks bundle run recommend_pitches --params series_start=2026-09-25
databricks bundle run recommend_pitches --params opponent_id=116,series_start=2027-04-01,lineup=<ids>
```

## Which series, lineup and pitchers

- **Series** (`recommender/inputs.py`, `choose_series`):
  - **scheduled:** the next Pirates series starting within `max_days_ahead` (10) days.
  - **demo:** with none that close (the offseason), the last series already played.
  - **chosen / custom:** `series_start` picks a scheduled series. With `opponent_id` and a
    date that isn't on the schedule, it's a custom series, for a future game.
- **Lineup:** teams post lineups only a few hours before a game, so by default it's
  projected. It's the opponent's batters with at least 3 starts in its last 10 games, at
  their most common batting order spot. `lineup` (batter ids in order) overrides it.
- **Pitchers:** everyone who pitched for the Pirates in their last 15 games, plus the
  series' probable starters. Position players who pitched in a blowout are left out.
  `pitchers` overrides the list.
- **No look-ahead:** everything is computed from data before the series starts. The
  feature lookups use the series start as the game date, so a demo on a past series sees
  what a real run that morning would have.

## Options and scenarios

- **Pitch types:** from the pitcher's latest `pitcher_arsenal_features` row, limited to
  pitches with at least 3% usage and 25 pitches. The hypothetical pitch has his real
  average velocity, spin, movement, release and extension for that pitch type.
- **Zones:** Statcast zones 1–9 (the strike zone in thirds) and 11–14 (outside, by
  quadrant), scaled to the batter's own strike zone.
- **Scenarios** (`recommender/options.py`):

  | Scenario | Count | Outs | Runners |
  |---|---|---|---|
  | first_pitch | 0-0 | 1 | none |
  | ahead_0_2, ahead_1_2 | 0-2, 1-2 | 1 | none |
  | behind_2_0, behind_3_1 | 2-0, 3-1 | 1 | none |
  | full_count | 3-2 | 1 | none |
  | need_strikeout | 1-1 | 1 | third |
  | risp_two_outs | 1-1 | 2 | second |

  Each is a tie in the 5th inning, the second time through the order.

## Aim and miss

A pitcher aims somewhere, and the pitch lands somewhere around it. The recommender models
both:

1. **Landing spots.** The models score each pitch type on a regular grid of landing spots:
   13 across the plate area (every 0.25 ft from −1.5 to 1.5) by 17 heights (every 0.125
   of the batter's zone, from −0.5 to 1.5). That gives the expected run value of a pitch
   that lands there.
2. **Aim points.** Each zone has nine aim points: a 3×3 grid inside zones 1–9, and for
   zones 11–14, three beside the zone, three above or below it, and three off the corner.
3. **Miss.** A pitch aimed at a point lands around it with a normal miss of 0.65 ft in
   each direction (`miss_sd_ft`). The aim point's value is the average of the landing
   values, weighted by where it lands. Spots off the grid count as the nearest edge spot,
   which is an obvious ball.
4. **Option value.** A zone's value is the average of its nine aim points. The range
   (`expected_run_value_low` / `_high`) shows how much it matters where in the zone he
   aims.

**Why model the miss?** An earlier version scored each option as if the pitch landed in
the zone for sure. That made corners look best ("paint the low inside corner at 3-1"),
because they're the most valuable spots when hit and the version ignored what a miss
costs. It also made every option look 0.04–0.10 runs per pitch better than the pitcher's
real mix, which includes his misses.

**Where 0.65 ft comes from.** It's a rough league-wide value, not fitted per pitcher. With
it, a pitch aimed at the middle of the zone lands in the zone 68% of the time. On 3-1, when
pitchers most want a strike, 63% of real pitches land in the zone, which fits pitchers
aiming a little off the middle. A pitcher's scatter of locations can't stand in for his
command, because it mixes intended variety with misses.

## Compared with the pitcher's usual choice

- **`usual_share`:** how often the pitcher throws this pitch type into this zone in this
  count, against same- or opposite-side hitters, over the last 365 days. It's context for
  the report: whether the best options are already his habit or a change.
- **`usual_expected_run_value`:** his usual mix, valued at the landing spots of his real
  pitches with the same model scores. `vs_usual` is the option's value minus this. Both
  sides include misses: the usual mix through real locations, the option through the miss
  model.
- **Exact count or count group:** the mix is by exact count when the pitcher has at least
  40 pitches there against that side (`usual_by_exact_count`), and by count group
  otherwise. Pooling a whole group would be misleading: full count shares "two strikes"
  with 0-2, where pitchers throw far more chase pitches.

## Output: `pitch_option_scores`

One row per team × opponent × series start × pitcher × batter × scenario × pitch type ×
zone. A rerun replaces that series' rows (`replaceWhere`).

| Column | Meaning |
|---|---|
| `target_mode`, `lineup_source` | how the series and lineup were chosen |
| `expected_run_value`, `_low`, `_high` | the option's value and its range over aim points |
| `p_ball` … `p_home_run` | probability of each of the ten pitch results |
| `option_rank` | rank within the pitcher × batter × scenario (1 = lowest run value) |
| `usual_share` | how often he throws this pitch type into this zone in this count |
| `usual_expected_run_value`, `vs_usual` | his usual mix's value and the option's difference |
| `usual_share_covered` | share of his usual mix in pitch types that were scored |
| `pitch_usage_rate`, `pitcher_pitches_thrown` | how established the pitch is (trailing 365 days) |
| `tier` | strong / fine / avoid, relative to his usual mix (below) |
| `pitch_name`, `location`, `grid_x`, `grid_y` | labels for reports: "Slider", "chase low and in", and a 5×5 grid cell from the batter's point of view |
| `opponent_name`, `scenario_label`, `scenario_order` | labels for reports |

**Tiers** (`options.STRONG_MARGIN`, `AVOID_MARGIN`) are judgment calls, relative to the
pitcher's usual mix in the same situation:
- **strong:** at least 0.5 runs per 100 pitches better
- **avoid:** at least 1 run per 100 pitches worse
- **fine:** in between

The report groups options this way instead of ranking options that are within noise of
each other. On the Detroit demo, 4–23% of options are strong and 27–72% avoid, depending on
the situation.

**Locations** are from the batter's point of view. Zones are numbered from the catcher's
view, and a right-handed batter stands on the catcher's left. So zone 1 is "up and in" to a
right-handed batter and "up and away" to a left-handed one.

## Dashboard: Series prep

An AI/BI dashboard (`resources/series_prep.dashboard.yml`,
`src/dashboards/series_prep.lvdash.json`) reads `pitch_option_scores`. It's deployed with
the bundle, and its datasets point at the target's ml schema (`dataset_catalog` /
`dataset_schema`). It answers the questions in
[requirements_pitching_coach.md](requirements_pitching_coach.md):

- **Pitch plan** (pick the series, hitter and pitcher):
  - the top three options per situation, with tier, plain pitch and location, runs per
    100 pitches better than his usual mix, their range, and how often he throws that
    pitch there
  - a location chart for one situation: the best pitch aimed at each spot, from the
    hitter's point of view
  - his three most common choices per situation, and how they rate
- **Staff matchups** (pick the series and hitter):
  - each Pirates pitcher's usual mix against the hitter, in runs per 100 pitches better
    than the staff average, by situation and overall

The hitter and pitcher filters default to Riley Greene and Paul Skenes for the demo. A
later series needs new defaults, or none.

## Results: demo on the Pirates at Detroit, 2026-09-25

With no upcoming series in November, the job fell back to a demo on the last series of
2026. It used data through 2026-09-24 and scored:
- 14 Detroit batters (the projected lineup) against 15 Pirates pitchers
- 8 scenarios, with each pitcher's 2–7 pitch types aimed at 13 zones
- 1,608,880 landing-spot rows, giving 94,640 options

**Where the best option is aimed, by scenario** (averages over all pitcher × batter pairs):

| Scenario | Usual mix | Best option | Gain | Gap to 2nd | Best is chase / corner / middle |
|---|---|---|---|---|---|
| first_pitch | −0.001 | −0.008 | −0.007 | 0.001 | 0% / 3% / 97% |
| ahead_0_2 | −0.007 | −0.014 | −0.007 | 0.003 | 80% / 10% / 10% |
| ahead_1_2 | −0.002 | −0.011 | −0.009 | 0.003 | 69% / 14% / 18% |
| behind_2_0 | −0.001 | −0.011 | −0.011 | 0.002 | 0% / 1% / 99% |
| behind_3_1 | −0.004 | −0.021 | −0.017 | 0.003 | 0% / 3% / 97% |
| full_count | +0.006 | −0.028 | −0.034 | 0.004 | 7% / 31% / 62% |
| need_strikeout | −0.007 | −0.021 | −0.014 | 0.003 | 42% / 9% / 49% |
| risp_two_outs | −0.008 | −0.018 | −0.010 | 0.002 | 16% / 21% / 63% |

Values are runs per pitch; negative is good for the pitcher. "Middle" is zones 2, 4, 5, 6
and 8; "corner" is 1, 3, 7 and 9.

What the numbers say:
- **The advice follows the count.** Behind in the count, aim at the middle. Ahead,
  expand: at 0-2 the best option is a chase pitch 80% of the time.
- **Effect of the miss model:** before it, the best option was a corner most of the time
  in every count. It also beat the usual mix by 0.04–0.10 runs per pitch. With misses on
  both sides, the gains are 0.007–0.034, largest at full count.
- **Close calls:** the best and second-best options differ by 0.001–0.004 runs. The best
  option's own range across aim points (0.002–0.014) is usually wider than that gap.

**Example: Paul Skenes against Riley Greene (left-handed)**

| Scenario | Best options (pitch type, zone) | His most common choices |
|---|---|---|
| 0-2 | slider 14, splitter 13, sweeper 14 | fastball 11 (27%), changeup 13 (15%), slider 14 (11%) |
| 3-1 | sinker 5, slider 5, splitter 5 | fastball 11 (14%), fastball 4 (7%), fastball 1 (7%) |

Zones are from the catcher's view, and a left-handed batter stands on the catcher's
right. So for Greene, 14 is low and in (a back-foot slider), 13 low and away, and 11 up and
away.
- **0-2:** his third most common choice is also the model's best.
- **3-1:** the model prefers aiming a sinker or slider at the middle over his habit of
  fastballs up and away, by about 0.02 runs per pitch.

## Backtest: does a better-rated pitch give up fewer runs?

The recommender picks among options **within one situation**, so that's where the models
have to be right. The earlier decile check on 2026 could look good just because the count
drives both prediction and result. The `backtest_recommender` task of
`train_outcome_models` (`src/databucs/models/backtest.py`) compares each 2026 pitch with
the other pitches the same pitcher threw in the same count, outs and runners. The models
never trained on 2026.

| Compared within | Pitches | Slope (±SE) | Real gap, worst vs best third | Expected gap |
|---|---|---|---|---|
| pitcher × count × outs × runners | 379,954 | 1.05 ± 0.01 | 7.0 ± 0.1 runs/100 | 6.4 |
| … and in or out of the zone | 258,211 | 0.96 ± 0.02 | 4.1 ± 0.1 runs/100 | 4.2 |

- **Slope:** the regression of actual run value on expected run value, both measured
  relative to the situation's average. A slope of 1 means the differences the models see
  between pitches in the same situation show up in the real results, at the same size.
  - It's 1.05, so they do.
  - Within the same situation and the same side of the zone edge it's still 0.96. So the
    models tell pitches apart beyond ball versus strike.
- **Gap:** in the same situation, the pitches the models rated best (top third) gave up 7.0
  fewer runs per 100 pitches than the worst third. The models expected 6.4.
- **By count group:**
  - hitter ahead: slope 1.24 (53k pitches)
  - neutral: 1.07 (227k)
  - two strikes: 0.95 (100k)

  When the hitter is ahead, real differences are larger than predicted. So if anything,
  the models understate how much the pitch choice matters there.
- **Coverage:** only situations where a pitcher threw at least 15 pitches count, which is
  53% of 2026 pitches. Standard errors come from 20 folds of games.

**What it doesn't test:**
- It scores pitches where they landed, not where they were aimed, so the aim-and-miss
  model and the tiers aren't tested directly.
- It's built from pitches pitchers chose to throw (selection bias). A pitch the models rate
  well in a situation where pitchers rarely throw it is less tested than these numbers
  suggest.
- A direct test of the tiers would need the full recommender for every 2026 situation:
  about 23 million scoring rows for the Pirates alone, too heavy for Free Edition.

## Known limitations

- **Miss model:** one league-wide miss size for every pitcher and pitch type. Real command
  differs, both by pitcher and by pitch (breaking balls miss more). A pitcher-specific
  estimate needs intended targets, which public data doesn't have.
- **Selection bias:** the models learn from pitches pitchers chose to throw. See the
  outcome model docs.
- **No sequencing:** every option is scored on its own, without the previous pitches or
  how predictable the choice is. The plan's next-pitch model is meant to cover that.
- **Close calls:** the top options are often within a few thousandths of a run of each
  other, and within each other's ranges. The report should show tiers, not a strict 1-to-N
  order.
- **Fixed situations:** inning, score and times through the order are fixed per scenario.
  Late and close, and times through the order for starters, aren't scored separately yet.
