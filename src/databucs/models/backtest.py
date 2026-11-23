"""Within-situation backtest: in the same situation, do pitches the models rate better give
up fewer runs in reality?

The recommender compares options within one situation (a pitcher, a count, outs and
runners), so that's where the models have to be right. Overall calibration by decile can
look good just because the count drives both the prediction and the result. Here each
test-season pitch is compared with the other pitches the same pitcher threw in the same
situation:

- slope: regression of actual run value on expected run value, both relative to their
  situation's average. 1 means the differences the models see between pitches in the same
  situation show up in the real results, at the same size.
- best vs worst third: real run value of the pitches the models rated worst in their
  situation minus the best, per 100 pitches, next to what the models expected.

Two groupings: the situation, and the situation split into pitches in and out of the
zone (do the models tell pitches apart beyond ball versus strike?). Only situations with
at least MIN_PITCHES pitches count. Standard errors come from FOLDS folds of games.

Caveat: these are pitches as thrown, scored where they landed, from pitches pitchers chose
to throw. It tests the models' ranking within a situation, not the aim-and-miss model.
"""

from __future__ import annotations

MIN_PITCHES = 15
FOLDS = 20

GROUPINGS = {
    "situation": ["pitcher_id", "balls", "strikes", "outs", "base_state"],
    "situation_and_zone": ["pitcher_id", "balls", "strikes", "outs", "base_state", "is_in_zone"],
}


def backtest_sql(predictions: str, fact_pitch: str) -> str:
    """Per grouping and fold: slope, best vs worst third (actual and expected, runs per
    100 pitches) and pitches. `predictions` is the evaluation job's output table."""
    parts = []
    for name, keys in GROUPINGS.items():
        partition = ", ".join(keys)
        parts.append(f"""
  SELECT *, '{name}' AS grouping,
    count(*) OVER w AS n,
    run_value - avg(run_value) OVER w AS dy,
    expected_run_value - avg(expected_run_value) OVER w AS dx,
    ntile(3) OVER (PARTITION BY {partition} ORDER BY expected_run_value) AS third
  FROM p
  WINDOW w AS (PARTITION BY {partition})""")
    grouped = "\n  UNION ALL".join(parts)
    return f"""
WITH p AS (
  SELECT t.game_pk, t.pitcher_id, t.balls, t.strikes, t.outs, t.base_state, t.run_value,
    t.expected_run_value, f.is_in_zone, pmod(t.game_pk, {FOLDS}) AS fold
  FROM {predictions} t
  JOIN {fact_pitch} f USING (game_pk, at_bat_number, pitch_number)
  WHERE t.run_value IS NOT NULL
),
g AS ({grouped}
)
SELECT grouping, fold,
  count(*) AS pitches,
  sum(dx * dy) / sum(dx * dx) AS slope,
  100 * (avg(CASE WHEN third = 3 THEN run_value END)
    - avg(CASE WHEN third = 1 THEN run_value END)) AS worst_minus_best_actual,
  100 * (avg(CASE WHEN third = 3 THEN expected_run_value END)
    - avg(CASE WHEN third = 1 THEN expected_run_value END)) AS worst_minus_best_expected
FROM g
WHERE n >= {MIN_PITCHES}
GROUP BY ALL
"""


def summarize(folds):
    """Mean and standard error over folds, per grouping. `folds` is a pandas frame from
    backtest_sql."""
    import numpy as np

    metrics = ["slope", "worst_minus_best_actual", "worst_minus_best_expected"]
    rows = []
    for grouping, frame in folds.groupby("grouping"):
        row = {"grouping": grouping, "pitches": int(frame["pitches"].sum())}
        for m in metrics:
            values = frame[m].astype(float)
            row[m] = float(values.mean())
            row[f"{m}_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(row)
    return rows
