"""Expected run value of a pitch, from the five outcome models.

Each competitive pitch ends in one of ten results. Their probabilities come from
multiplying along the outcome tree:

  ball, called_strike, hit_by_pitch   P(take)  x P(result | take)
  whiff                               P(swing) x P(whiff | swing)
  foul                                P(swing) x P(contact | swing) x P(foul | contact)
  out, single, ..., home_run          P(swing) x P(contact | swing) x P(in play | contact)
                                      x P(result | in play)

Each result's run value in a given state (balls, strikes, outs, base_state) is
"runs scored plus run expectancy after the pitch" minus "run expectancy before it",
averaged over the training seasons only:

- Results that change the count are averaged per full state, because the count decides
  what they do (a ball at 3-1 is a walk). A called strike, a whiff and a foul with fewer
  than two strikes all just add a strike, so they share one value; a foul with two strikes
  changes nothing and gets its own. Pooling them gives every state plenty of data, even
  3-0, where batters rarely swing. (It ignores the rare dropped third strike.)
- Results that end the plate appearance (hit by pitch and the five in-play results) don't
  depend on the count once they happen: a bases-loaded home run scores four whatever the
  count. Their "after" value is averaged per (outs, base_state, result), which has plenty of
  data even for rare combinations, and the count only enters through the run expectancy
  before the pitch. (This ignores that runners going on the pitch at 3-2 with two outs
  advance a little further on hits.)

Expected run value is the sum of probability x value over the ten results. As with
run_value, positive is good for the batter.
"""

from __future__ import annotations

from databucs.models.outcomes import BY_NAME, COMPETITIVE

RESULTS = [
    "ball",
    "called_strike",
    "hit_by_pitch",
    "whiff",
    "foul",
    "out",
    "single",
    "double",
    "triple",
    "home_run",
]

COUNT_RESULTS = ["ball", "called_strike", "whiff", "foul"]
PA_ENDING_RESULTS = [r for r in RESULTS if r not in COUNT_RESULTS]

RESULT_SQL = """
CASE
  WHEN NOT is_swing THEN
    CASE description
      WHEN 'called_strike' THEN 'called_strike'
      WHEN 'hit_by_pitch' THEN 'hit_by_pitch'
      ELSE 'ball'
    END
  WHEN is_whiff OR description = 'foul_tip' THEN 'whiff'
  WHEN NOT is_in_play THEN 'foul'
  WHEN events IN ('single', 'double', 'triple', 'home_run') THEN events
  ELSE 'out'
END
"""

# Probability of each result as a product of model outputs. `p.<model>.<class>` refers to
# the prediction struct from each model.
RESULT_PROBABILITY_SQL = {
    "ball": "p.swing.p_take * p.take_outcome.p_ball",
    "called_strike": "p.swing.p_take * p.take_outcome.p_called_strike",
    "hit_by_pitch": "p.swing.p_take * p.take_outcome.p_hit_by_pitch",
    "whiff": "p.swing.p_swing * p.whiff.p_whiff",
    "foul": "p.swing.p_swing * p.whiff.p_contact * p.contact_outcome.p_foul",
    **{
        result: (
            "p.swing.p_swing * p.whiff.p_contact * p.contact_outcome.p_in_play"
            f" * p.in_play_outcome.p_{result}"
        )
        for result in ("out", "single", "double", "triple", "home_run")
    },
}


def run_values_sql(fact_pitch: str, seasons: list[int]) -> str:
    """Run value per (balls, strikes, outs, base_state, result) over `seasons` (see the
    module docstring). Walk-off pitches have no run_value and are left out."""
    season_list = ", ".join(str(s) for s in seasons)
    count_results = ", ".join(f"'{r}'" for r in COUNT_RESULTS)
    return f"""
WITH pitches AS (
  SELECT
    balls, strikes, outs, base_state,
    {RESULT_SQL} AS result,
    run_expectancy_before,
    -- runs scored on the pitch + run expectancy after it
    run_value + run_expectancy_before AS value_after
  FROM {fact_pitch}
  WHERE ({COMPETITIVE}) AND season IN ({season_list}) AND run_value IS NOT NULL
),
states AS (
  SELECT balls, strikes, outs, base_state, avg(run_expectancy_before) AS value_before
  FROM pitches
  GROUP BY ALL
),
-- What a count-changing result does: add a ball, add a strike, or (a foul with two
-- strikes) nothing
transitions AS (
  SELECT
    balls, strikes, outs, base_state, value_after,
    CASE
      WHEN result = 'ball' THEN 'ball'
      WHEN result = 'foul' AND strikes = 2 THEN 'foul_two_strikes'
      ELSE 'strike'
    END AS transition
  FROM pitches
  WHERE result IN ({count_results})
),
transition_values AS (
  SELECT balls, strikes, outs, base_state, transition, count(*) AS pitches,
    avg(value_after) AS value_after
  FROM transitions
  GROUP BY ALL
),
count_results AS (
  SELECT
    s.*,
    r.result,
    CASE
      WHEN r.result = 'ball' THEN 'ball'
      WHEN r.result = 'foul' AND s.strikes = 2 THEN 'foul_two_strikes'
      ELSE 'strike'
    END AS transition
  FROM states s
  CROSS JOIN (SELECT explode(array({count_results})) AS result) r
),
pa_ending_results AS (
  SELECT outs, base_state, result, count(*) AS pitches, avg(value_after) AS value_after
  FROM pitches
  WHERE result NOT IN ({count_results})
  GROUP BY ALL
)
SELECT c.balls, c.strikes, c.outs, c.base_state, c.result, t.pitches,
  t.value_after - c.value_before AS run_value
FROM count_results c
JOIN transition_values t USING (balls, strikes, outs, base_state, transition)
UNION ALL
SELECT s.balls, s.strikes, s.outs, s.base_state, r.result, r.pitches,
  r.value_after - s.value_before AS run_value
FROM states s
JOIN pa_ending_results r USING (outs, base_state)
"""


def prediction_struct_type(model_name: str) -> str:
    """Spark DDL for one model's pyfunc output (one double per class)."""
    classes = BY_NAME[model_name].classes
    return "struct<" + ", ".join(f"p_{c}: double" for c in classes) + ">"
