"""Exact plate-appearance simulator: play a plate appearance count by count under a pitch
policy, and add up what happens.

A plate appearance has only twelve counts, and each pitch moves to a later count or ends
it, so there's no need to sample: work backwards from 3-2 to 0-0. At each count a policy
picks a pitch (or a mix of pitches), whose result probabilities come from the outcome
models (via the batch table, pitch_option_scores):

  ball            next ball, or a walk with three balls
  called strike,  next strike, or a strikeout with two strikes
  whiff
  foul            next strike, or the same count with two strikes
  hit by pitch,   the plate appearance ends
  in-play results

A foul with two strikes repeats the count, so at two strikes the value V of a pitch
choice solves V = (sum over other results) + p_foul * V, i.e. it's divided by 1 - p_foul.

What it adds up, for each count to the end of the plate appearance:
- run value: the sum of the pitches' run values (the same per-result run values as the
  recommender, relative to the run expectancy at that count; positive is good for the
  batter). From 0-0 it's the plate appearance's runs above average.
- how it ends: strikeout, walk, hit by pitch, an out in play, single, ..., home run
- pitches thrown

Policies:
- usual: the pitcher's usual mix at each count (as the recommender values it)
- recommended: the top option by one-pitch expected run value, the report's rank 1
- best: the option with the lowest run value to the end of the plate appearance, using
  his own later counts rather than the league's run expectancy (dynamic programming)

All three assume the batter doesn't adjust to what the pitcher does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from databucs.models.run_values import RESULTS

COUNTS = [(b, s) for b in range(4) for s in range(3)]
# Later counts first, so a count's next counts are always solved before it
SOLVE_ORDER = sorted(COUNTS, key=lambda c: -(c[0] + c[1]))
ENDINGS = [
    "strikeout",
    "walk",
    "hit_by_pitch",
    "out",
    "single",
    "double",
    "triple",
    "home_run",
]
POLICIES = ["usual", "recommended", "best"]


def next_state(balls: int, strikes: int, result: str) -> tuple[int, int] | str:
    """The count after a pitch, or how the plate appearance ended."""
    if result == "ball":
        return "walk" if balls == 3 else (balls + 1, strikes)
    if result in ("called_strike", "whiff"):
        return "strikeout" if strikes == 2 else (balls, strikes + 1)
    if result == "foul":
        return (balls, strikes) if strikes == 2 else (balls, strikes + 1)
    return result  # hit_by_pitch and the in-play results end it


@dataclass
class Choice:
    """One pitch choice at a count: each result's probability (RESULTS order)."""

    p: np.ndarray
    label: object = None  # e.g. (pitch_type, zone), for reporting what was picked


@dataclass
class Outcome:
    """What follows from a count under a policy, to the end of the plate appearance."""

    run_value: float
    endings: np.ndarray  # ENDINGS order
    pitches: float
    label: object = None


@dataclass
class PlateAppearance:
    policy: str
    by_count: dict[tuple[int, int], Outcome] = field(default_factory=dict)

    @property
    def start(self) -> Outcome:
        return self.by_count[(0, 0)]


def _evaluate(
    count: tuple[int, int],
    choice: Choice,
    rv: np.ndarray,
    solved: dict[tuple[int, int], Outcome],
) -> Outcome:
    balls, strikes = count
    value, pitches = float(choice.p @ rv), 1.0
    endings = np.zeros(len(ENDINGS))
    repeat = 0.0
    for i, result in enumerate(RESULTS):
        p = choice.p[i]
        nxt = next_state(balls, strikes, result)
        if nxt == count:
            repeat += p
        elif isinstance(nxt, tuple):
            later = solved[nxt]
            value += p * later.run_value
            endings += p * later.endings
            pitches += p * later.pitches
        else:
            endings[ENDINGS.index(nxt)] += p
    keep = 1.0 - repeat
    if keep < 1e-9:
        raise ValueError(f"a choice at {count} never ends the plate appearance")
    return Outcome(value / keep, endings / keep, pitches / keep, choice.label)


def simulate(
    policy: str,
    choices: dict[tuple[int, int], list[Choice]],
    run_values: dict[tuple[int, int], np.ndarray],
) -> PlateAppearance:
    """Solve every count under `policy`.

    choices: per count, the candidate choices. For "usual" that's one choice, his mix;
    for "recommended" and "best", the options. run_values: per count, each result's run
    value (RESULTS order) in the starting outs and base state.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    solved: dict[tuple[int, int], Outcome] = {}
    for count in SOLVE_ORDER:
        candidates, rv = choices[count], run_values[count]
        if not candidates:
            raise ValueError(f"no choices at {count}")
        if policy == "recommended":
            one_pitch = [float(c.p @ rv) for c in candidates]
            candidates = [candidates[int(np.argmin(one_pitch))]]
        elif policy == "usual" and len(candidates) != 1:
            raise ValueError("the usual policy takes one choice per count, his mix")
        outcomes = [_evaluate(count, c, rv, solved) for c in candidates]
        solved[count] = min(outcomes, key=lambda o: o.run_value)
    return PlateAppearance(policy, solved)


def simulate_matchups(options, run_values: dict[tuple[int, int], np.ndarray]):
    """One row per (pitcher, batter, policy) from the batch table's options.

    options: a pandas frame with pitcher_id, batter_id, balls, strikes, pitch_type, zone,
    p_<result> and usual_p_<result>, covering every count for each pair. The usual policy
    is left out for a pair if he has no usual mix in some count.
    """
    import pandas as pd

    p_columns = [f"p_{r}" for r in RESULTS]
    usual_columns = [f"usual_p_{r}" for r in RESULTS]
    rows = []
    for (pitcher_id, batter_id), pair in options.groupby(["pitcher_id", "batter_id"]):
        by_count = {c: g for c, g in pair.groupby(["balls", "strikes"])}
        missing = set(COUNTS) - set(by_count)
        if missing:
            raise ValueError(f"pitcher {pitcher_id} vs {batter_id}: no options at {missing}")
        option_choices = {
            c: [
                Choice(p, (t, int(z)))
                for p, t, z in zip(g[p_columns].to_numpy(), g["pitch_type"], g["zone"], strict=True)
            ]
            for c, g in by_count.items()
        }
        usual = {c: g[usual_columns].iloc[0].to_numpy(dtype=float) for c, g in by_count.items()}
        policies = {"recommended": option_choices, "best": option_choices}
        if not any(np.isnan(p).any() for p in usual.values()):
            policies = {"usual": {c: [Choice(p)] for c, p in usual.items()}, **policies}
        for policy, choices in policies.items():
            start = simulate(policy, choices, run_values).start
            first_type, first_zone = start.label or (None, None)
            rows.append(
                {
                    "pitcher_id": pitcher_id,
                    "batter_id": batter_id,
                    "policy": policy,
                    "run_value": start.run_value,
                    **{f"p_{e}": float(p) for e, p in zip(ENDINGS, start.endings, strict=True)},
                    "pitches": start.pitches,
                    "first_pitch_type": first_type,
                    "first_pitch_zone": first_zone,
                }
            )
    return pd.DataFrame(rows)
