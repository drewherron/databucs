# Plate-appearance simulator and matchup rankings

The recommender values one pitch at a time. The simulator plays out whole plate
appearances, count by count, so it can say how each Pirates pitcher fares against each
batter: runs per plate appearance, strikeout and walk rates, and pitches thrown. Averaged
over a lineup, that ranks the staff for a series.

It runs as the `simulate_matchups` task after `recommend_pitches` in the same job, and
writes `<ml schema>.matchup_simulations`.

## How it works

`recommender/simulate.py`:

- **Exact, not sampled.** A plate appearance has twelve counts, and each pitch either moves
  to a later count or ends it. Working backwards from 3-2 to 0-0 gives exact expected
  values with no random noise.
- **Transitions:** a ball adds a ball (a walk at three balls); a called strike or whiff
  adds a strike (a strikeout at two strikes); a foul adds a strike, or repeats the count
  with two strikes; a hit by pitch or a ball in play ends it. A repeated two-strike count
  is solved exactly: the value is divided by 1 − P(foul).
- **Inputs** come from the batch table, so the simulator and the report value pitches the
  same way:
  - each option's result probabilities (`p_<result>`), at every count
  - the pitcher's usual mix's result probabilities (`usual_p_<result>`), at every count
  - the run value of each result at each count (the recommender's run values)

  To cover all twelve counts, the bases-empty, one-out scenarios now include every count
  (14 scenarios in all). The job checks that each option's probabilities and run values
  give back the table's expected run value (largest gap on the demo: 8e-17).
- **Run value** of a plate appearance is the sum of its pitches' run values: runs above
  average from 0-0, positive good for the batter. It's relative to the run expectancy
  before the plate appearance, so it reads as "runs per plate appearance better or worse
  than an average matchup".

### Policies

| Policy | At each count the pitcher throws |
|---|---|
| `usual` | his usual mix there, at his real landing spots |
| `recommended` | the report's top option (lowest one-pitch expected run value) |
| `best` | the option with the lowest run value over the rest of the plate appearance, using his own later counts rather than league run expectancy (dynamic programming) |

## Output: `matchup_simulations`

One row per team × opponent × series start × pitcher × batter × policy, with the batch
table's labels (names, lineup spot, sides, probable starter) and:

| Column | Meaning |
|---|---|
| `run_value` | expected runs above average for the plate appearance |
| `p_strikeout`, `p_walk`, `p_hit_by_pitch`, `p_out`, `p_single`, `p_double`, `p_triple`, `p_home_run` | how it ends; they sum to 1 |
| `pitches` | expected pitches thrown |
| `first_pitch_type`, `first_pitch_zone` | the policy's 0-0 choice (empty for `usual`) |
| `outs`, `base_state` | the starting state (1 out, bases empty) |

## Results: demo on the Pirates at Detroit, 2026-09-25

15 pitchers × 14 batters × 3 policies (630 rows). Averages over the projected lineup, per
100 plate appearances:

| Pitcher | Runs (usual) | K% | BB% | Pitches/PA | Gain, recommended | Gain, best |
|---|---|---|---|---|---|---|
| Jared Jones | −1.75 | 26.9 | 8.6 | 3.86 | 2.88 | 3.15 |
| Mason Montgomery | −1.49 | 27.5 | 8.4 | 3.85 | 3.10 | 3.36 |
| Bubba Chandler | −1.39 | 23.6 | 8.0 | 3.76 | 2.73 | 2.88 |
| Wilber Dotel | −1.24 | 23.8 | 7.3 | 3.78 | 2.68 | 2.86 |
| Paul Skenes | −1.01 | 22.9 | 7.9 | 3.87 | 2.21 | 2.40 |
| … | | | | | | |
| Brandon Eisert | +1.36 | 21.9 | 12.4 | 3.87 | 3.62 | 3.73 |
| Kirby Yates | +1.40 | 23.4 | 11.4 | 3.99 | 2.14 | 2.21 |

**Checked against what they really did in 2026** (13 pitchers with at least 100 plate
appearances before the series; real rates are against all batters, simulated against
Detroit's lineup):

| | Correlation | Spread (sd), simulated | Spread, real | Mean, simulated | Mean, real |
|---|---|---|---|---|---|
| K% | 0.71 | 2.0 | 4.8 | 23.7 | 26.4 |
| BB% | 0.49 | 2.3 | 1.9 | 9.8 | 9.3 |

What it shows:
- **The order is mostly right but the spread is too narrow, especially for strikeouts.**
  Montgomery strikes out 37.8% in reality and 27.5% in the simulation; Skenes 28.0% and
  22.9%. The outcome models describe a pitch by its traits (velocity, movement, release),
  not by who throws it, so pitchers whose stuff plays above its measurements get pulled
  toward the average. That's why Skenes ranks fifth.
- **Following the report's top option every pitch gains 2–5 runs per 100 plate
  appearances** over the usual mix. That's an upper bound: options assume a league-average
  miss while the usual mix is valued at where his pitches really landed (see
  [recommender.md](recommender.md)), and a pitcher who throws the same thing every time in
  a count would be predictable, which the models can't see.
- **One pitch at a time is nearly as good as planning the whole plate appearance:**
  `best` adds only 0.1–0.3 runs per 100 over `recommended`. Run expectancy by count
  already prices in what comes later, so the report's per-pitch ranking doesn't need a
  plan to be useful.
- Wilder relievers (Soto, Ramírez, Doval, Curtis) get the biggest gains: their usual mix
  walks 12–14% of these batters, and the recommended options throw more strikes.

## Limitations

- **Compressed talent:** see above. Rankings are better read as an order than as sizes.
  Adding the pitcher's own past results (or a pitcher effect) would widen the spread.
- **One starting state:** one out, bases empty, a mid-game tie the second time through.
  Runners change both the run values and pitch choices.
- **No adaptation:** the batter doesn't adjust to a predictable pitcher, and the pitcher
  doesn't tire. Times through the order is fixed at two.
- **Lineup average:** each batter counts the same; a starter sees the top of the order
  more often.

## Running it

```bash
databricks bundle run recommend_pitches                      # scores the next series, then simulates
databricks bundle run recommend_pitches --params series_start=2026-09-25
```

`simulate-matchups` alone takes `--opponent-id` and `--series-start`; by default it
simulates the series scored most recently.
