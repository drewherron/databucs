"""Forward-in-time test of the plate-appearance simulator: do pitchers' simulated
strikeout and walk rates, built only from data before a date, forecast what they do after
it?

Unlike comparing the simulation with the same season's results, this can't be passed by
feeding the models a pitcher's own results: those results are from before the date, and
the test scores what comes after.

- Pitchers: the busiest after the date, among those with enough plate appearances on both
  sides of it.
- Hitters: a fixed panel, the regulars with the most plate appearances before the date, so
  every pitcher faces the same hitters in the simulation.
- Real rates are against whoever they actually faced, so part of each gap is opponents.

Two references put the numbers in context:
- naive: his own rate before the date as the forecast. A forecast from the models that
  can't beat this isn't adding much, though the models' job is pitch choices, not this.
- noise: part of the real rates' spread is luck in a few hundred plate appearances. The
  "true" spread removes the binomial noise, var(observed) - mean(p(1 - p) / n), and is
  what a perfect forecast's spread would be.
"""

from __future__ import annotations

import math

import pandas as pd

STRIKEOUT_EVENTS = ("strikeout", "strikeout_double_play")
WALK_EVENTS = ("walk", "intent_walk")
RATES = {"k": STRIKEOUT_EVENTS, "bb": WALK_EVENTS}


def _rates(events: str) -> str:
    return ",\n    ".join(
        f"avg(CASE WHEN {events} IN ({', '.join(repr(e) for e in values)}) THEN 1 ELSE 0 END)"
        f" AS {name}_rate"
        for name, values in RATES.items()
    )


def pitcher_outcomes_sql(fact_pitch: str, as_of: str, end: str) -> str:
    """Per pitcher: plate appearances and K / BB rates before the date (that season) and
    after it, through `end`."""
    return f"""
WITH pa AS (
  SELECT pitcher_id, game_date >= DATE'{as_of}' AS after, events
  FROM {fact_pitch}
  WHERE events IS NOT NULL
    AND season = year(DATE'{as_of}')
    AND game_date <= DATE'{end}'
),
by_side AS (
  SELECT pitcher_id, after, count(*) AS pa,
    {_rates("events")}
  FROM pa
  GROUP BY ALL
)
SELECT a.pitcher_id,
  b.pa AS pa_before, b.k_rate AS k_rate_before, b.bb_rate AS bb_rate_before,
  a.pa AS pa_after, a.k_rate AS k_rate_after, a.bb_rate AS bb_rate_after
FROM by_side a
JOIN by_side b ON b.pitcher_id = a.pitcher_id AND NOT b.after
WHERE a.after
"""


def choose_pitchers(outcomes: pd.DataFrame, n: int, min_pa: int) -> list[int]:
    """The n pitchers with the most plate appearances after the date, among those with at
    least min_pa on each side of it."""
    eligible = outcomes[(outcomes["pa_before"] >= min_pa) & (outcomes["pa_after"] >= min_pa)]
    return [int(p) for p in eligible.nlargest(n, "pa_after")["pitcher_id"]]


def batter_panel_sql(fact_pitch: str, as_of: str, n: int) -> str:
    """The n hitters with the most plate appearances that season before the date."""
    return f"""
SELECT batter_id, count(*) AS pa
FROM {fact_pitch}
WHERE events IS NOT NULL
  AND season = year(DATE'{as_of}')
  AND game_date < DATE'{as_of}'
GROUP BY batter_id
ORDER BY pa DESC, batter_id
LIMIT {n}
"""


def true_sd(rates: pd.Series, pa: pd.Series) -> float:
    """Spread of the rates with binomial noise taken out (0 if noise explains it all)."""
    noise = float((rates * (1 - rates) / pa).mean())
    return math.sqrt(max(float(rates.var()) - noise, 0.0))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per rate (k, bb). `df` has one row per pitcher with sim_<rate>,
    <rate>_rate_before, <rate>_rate_after, pa_after."""
    rows = []
    for name in RATES:
        sim, before = df[f"sim_{name}"], df[f"{name}_rate_before"]
        after, pa = df[f"{name}_rate_after"], df["pa_after"]
        rows.append(
            {
                "rate": name,
                "pitchers": len(df),
                "corr_sim": float(sim.corr(after)),
                "corr_naive": float(before.corr(after)),
                "rmse_sim": float(((sim - after) ** 2).mean() ** 0.5),
                "rmse_naive": float(((before - after) ** 2).mean() ** 0.5),
                "mean_sim": float(sim.mean()),
                "mean_after": float(after.mean()),
                "sd_sim": float(sim.std()),
                "sd_after": float(after.std()),
                "sd_after_true": true_sd(after, pa),
            }
        )
    return pd.DataFrame(rows)
