import numpy as np
import pytest

from databucs.models.run_values import RESULTS
from databucs.recommender.simulate import (
    COUNTS,
    ENDINGS,
    Choice,
    next_state,
    simulate,
)


def always(result: str) -> Choice:
    p = np.zeros(len(RESULTS))
    p[RESULTS.index(result)] = 1.0
    return Choice(p, result)


def mix(**probabilities) -> Choice:
    p = np.array([probabilities.get(r, 0.0) for r in RESULTS])
    return Choice(p / p.sum())


ZERO_RV = {c: np.zeros(len(RESULTS)) for c in COUNTS}


@pytest.mark.parametrize(
    "count, result, expected",
    [
        ((0, 0), "ball", (1, 0)),
        ((3, 1), "ball", "walk"),
        ((1, 1), "called_strike", (1, 2)),
        ((1, 2), "whiff", "strikeout"),
        ((0, 1), "foul", (0, 2)),
        ((2, 2), "foul", (2, 2)),
        ((0, 0), "home_run", "home_run"),
        ((3, 2), "hit_by_pitch", "hit_by_pitch"),
    ],
)
def test_next_state(count, result, expected):
    assert next_state(*count, result) == expected


def test_four_balls_is_a_walk():
    pa = simulate("usual", {c: [always("ball")] for c in COUNTS}, ZERO_RV)
    assert pa.start.endings[ENDINGS.index("walk")] == pytest.approx(1.0)
    assert pa.start.pitches == pytest.approx(4.0)
    assert pa.by_count[(3, 2)].pitches == pytest.approx(1.0)


def test_three_strikes_is_a_strikeout():
    pa = simulate("usual", {c: [always("called_strike")] for c in COUNTS}, ZERO_RV)
    assert pa.start.endings[ENDINGS.index("strikeout")] == pytest.approx(1.0)
    assert pa.start.pitches == pytest.approx(3.0)


def test_two_strike_fouls_repeat_the_count():
    choices = {c: [mix(foul=0.5, out=0.5)] for c in COUNTS}
    pa = simulate("usual", choices, ZERO_RV)
    # From 0-2: one pitch, then a coin flip between another pitch and the end
    assert pa.by_count[(0, 2)].pitches == pytest.approx(2.0)
    assert pa.by_count[(0, 2)].endings[ENDINGS.index("out")] == pytest.approx(1.0)
    # From 0-0: two fouls to get to 0-2, unless the ball is put in play first
    assert pa.start.pitches == pytest.approx(0.5 * 1 + 0.25 * 2 + 0.25 * (2 + 2))


def random_choice(rng) -> Choice:
    return Choice(rng.dirichlet(np.ones(len(RESULTS))))


def test_run_values_add_up_to_the_change_in_run_expectancy():
    """With run values defined as "value after minus run expectancy before", the plate
    appearance's run value is the value of how it ended minus the run expectancy at 0-0,
    whatever the pitcher throws."""
    rng = np.random.default_rng(0)
    expectancy = {c: rng.normal(0.5, 0.1) for c in COUNTS}
    ending_value = dict(zip(ENDINGS, rng.normal(0.5, 0.5, len(ENDINGS)), strict=True))
    run_values = {}
    for c in COUNTS:
        after = [next_state(*c, r) for r in RESULTS]
        run_values[c] = (
            np.array([(expectancy[a] if isinstance(a, tuple) else ending_value[a]) for a in after])
            - expectancy[c]
        )
    choices = {c: [random_choice(rng)] for c in COUNTS}
    pa = simulate("usual", choices, run_values)
    assert pa.start.endings.sum() == pytest.approx(1.0)
    values = np.array([ending_value[e] for e in ENDINGS])
    assert pa.start.run_value == pytest.approx(pa.start.endings @ values - expectancy[(0, 0)])


def test_best_is_never_worse_than_recommended():
    rng = np.random.default_rng(1)
    run_values = {c: rng.normal(0, 0.2, len(RESULTS)) for c in COUNTS}
    choices = {c: [random_choice(rng) for _ in range(5)] for c in COUNTS}
    best = simulate("best", choices, run_values)
    recommended = simulate("recommended", choices, run_values)
    for c in COUNTS:
        assert best.by_count[c].run_value <= recommended.by_count[c].run_value + 1e-12


def test_recommended_takes_the_best_one_pitch_value():
    # A called strike is worth less to the batter than a ball everywhere, so it's picked
    run_values = {c: np.zeros(len(RESULTS)) for c in COUNTS}
    for c in COUNTS:
        run_values[c][RESULTS.index("foul")] = 0.01
        run_values[c][RESULTS.index("ball")] = 0.02
    run_values[(0, 2)][RESULTS.index("called_strike")] = -0.2
    choices = {c: [always("ball"), always("called_strike")] for c in COUNTS}
    pa = simulate("recommended", choices, run_values)
    assert pa.by_count[(0, 2)].label == "called_strike"
    assert pa.start.endings[ENDINGS.index("strikeout")] == pytest.approx(1.0)


def test_a_choice_that_never_ends_is_an_error():
    with pytest.raises(ValueError):
        simulate("usual", {c: [always("foul")] for c in COUNTS}, ZERO_RV)


def test_usual_needs_one_choice_per_count():
    with pytest.raises(ValueError):
        simulate("usual", {c: [always("ball"), always("foul")] for c in COUNTS}, ZERO_RV)


def options_frame(rng, pairs=((1, 10), (1, 11)), usual=True):
    import pandas as pd

    rows = []
    for pitcher_id, batter_id in pairs:
        for balls, strikes in COUNTS:
            mix_p = rng.dirichlet(np.ones(len(RESULTS)))
            for pitch_type in ("FF", "SL"):
                for zone in (5, 14):
                    p = rng.dirichlet(np.ones(len(RESULTS)))
                    rows.append(
                        {
                            "pitcher_id": pitcher_id,
                            "batter_id": batter_id,
                            "balls": balls,
                            "strikes": strikes,
                            "pitch_type": pitch_type,
                            "zone": zone,
                            **{f"p_{r}": x for r, x in zip(RESULTS, p, strict=True)},
                            **{
                                f"usual_p_{r}": (x if usual else np.nan)
                                for r, x in zip(RESULTS, mix_p, strict=True)
                            },
                        }
                    )
    return pd.DataFrame(rows)


def test_simulate_matchups_gives_each_pair_every_policy():
    from databucs.recommender.simulate import simulate_matchups

    rng = np.random.default_rng(2)
    run_values = {c: rng.normal(0, 0.1, len(RESULTS)) for c in COUNTS}
    out = simulate_matchups(options_frame(rng), run_values)
    assert len(out) == 2 * 3
    endings = out[[f"p_{e}" for e in ENDINGS]].sum(axis=1)
    assert endings.to_numpy() == pytest.approx(1.0)
    pivot = out.pivot(index="batter_id", columns="policy", values="run_value")
    assert (pivot["best"] <= pivot["recommended"] + 1e-12).all()
    assert out["first_pitch_type"].isin(["FF", "SL", None]).all()


def test_simulate_matchups_skips_usual_without_a_mix():
    from databucs.recommender.simulate import simulate_matchups

    rng = np.random.default_rng(3)
    run_values = {c: np.zeros(len(RESULTS)) for c in COUNTS}
    out = simulate_matchups(options_frame(rng, pairs=((1, 10),), usual=False), run_values)
    assert set(out["policy"]) == {"recommended", "best"}


def test_simulate_matchups_needs_every_count():
    from databucs.recommender.simulate import simulate_matchups

    rng = np.random.default_rng(4)
    frame = options_frame(rng, pairs=((1, 10),))
    frame = frame[~((frame.balls == 3) & (frame.strikes == 0))]
    with pytest.raises(ValueError):
        simulate_matchups(frame, {c: np.zeros(len(RESULTS)) for c in COUNTS})


def test_forward_test_summary():
    import pandas as pd

    from databucs.models import forward_test

    df = pd.DataFrame(
        {
            "sim_k": [0.20, 0.25, 0.30],
            "k_rate_before": [0.18, 0.26, 0.31],
            "k_rate_after": [0.19, 0.24, 0.33],
            "sim_bb": [0.08, 0.09, 0.10],
            "bb_rate_before": [0.07, 0.10, 0.09],
            "bb_rate_after": [0.08, 0.09, 0.11],
            "pa_after": [300, 300, 300],
        }
    )
    summary = forward_test.summarize(df).set_index("rate")
    assert summary.loc["k", "corr_sim"] > 0.9
    assert summary.loc["k", "pitchers"] == 3
    # The true spread is below the observed one: some of it is noise
    assert summary.loc["k", "sd_after_true"] < summary.loc["k", "sd_after"]


def test_true_sd_is_zero_when_noise_explains_the_spread():
    import pandas as pd

    from databucs.models.forward_test import true_sd

    assert true_sd(pd.Series([0.20, 0.22]), pd.Series([50, 50])) == 0.0


def test_choose_pitchers_needs_both_sides_of_the_date():
    import pandas as pd

    from databucs.models.forward_test import choose_pitchers

    outcomes = pd.DataFrame(
        {"pitcher_id": [1, 2, 3], "pa_before": [200, 50, 300], "pa_after": [250, 400, 160]}
    )
    assert choose_pitchers(outcomes, n=5, min_pa=150) == [1, 3]
    assert choose_pitchers(outcomes, n=1, min_pa=150) == [1]
