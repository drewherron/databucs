import numpy as np
import pandas as pd
import pytest

from databucs.features import batter_profile
from databucs.jobs.train_outcome_model import parse_args, should_promote, training_where
from databucs.models import evaluation, features
from databucs.models.outcomes import BY_NAME, MODELS


def looked_up_row(**overrides):
    """One row as it looks after the feature lookups: pitch and state columns + features."""
    row = {
        "pitch_type": "SL",
        "pitch_group": "breaking",
        "pitcher_throws": "R",
        "batter_stands": "R",
        "is_in_zone": False,
        "plate_x": 1.2,
        "plate_z": 1.0,
        "sz_top": 3.5,
        "sz_bot": 1.5,
        "pitch_velocity": 86.0,
        "pitch_spin_rate": 2500,
        "pitch_horizontal_break": -4.0,
        "pitch_induced_vertical_break": 2.0,
        "pitch_release_side": 2.0,
        "pitch_release_height": 5.8,
        "pitch_extension": 6.5,
        "pitch_arm_angle": 35.0,
        "balls": 1,
        "strikes": 2,
        "outs": 1,
        "inning": 3,
        "bat_score_diff": 0,
        "times_through_order": 1,
        "base_state": "1__",
    }
    for col in features.BATTER_FEATURES + features.BATTER_CELL_COUNTS:
        row[col] = 0.0
    for col in features.ARSENAL_FEATURES:
        row[col] = 0.0
    # Arsenal: this slider averages 85 mph, 11 mph slower than the fastball (so 96 mph)
    row.update(velocity=85.0, velocity_diff_from_fastball=-11.0)
    row.update(horizontal_break=-5.0, horizontal_break_diff_from_fastball=-20.0)
    row.update(induced_vertical_break=1.0, induced_vertical_break_diff_from_fastball=-15.0)
    row["breaking_chase_two_strikes_swing_rate"] = 0.37
    row["breaking_chase_two_strikes_pitches"] = 120
    row["breaking_zone_two_strikes_swing_rate"] = 0.9
    row.update(overrides)
    return row


def test_transform_returns_exactly_the_model_features_in_order():
    out = features.transform(pd.DataFrame([looked_up_row()]))
    assert list(out.columns) == features.FEATURES
    assert len(set(features.FEATURES)) == len(features.FEATURES)


def test_location_is_relative_to_the_batter():
    df = pd.DataFrame(
        [
            looked_up_row(batter_stands="R", plate_x=-1.0),
            looked_up_row(batter_stands="L", plate_x=1.0),
        ]
    )
    out = features.transform(df)
    # Both pitches are a foot toward the batter
    assert out["plate_x_toward_batter"].tolist() == [1.0, 1.0]
    # 1.0 ft is a quarter of a zone below the bottom (1.5 to 3.5)
    assert out["plate_height_in_zone"].iloc[0] == pytest.approx(-0.25)


def test_difference_from_the_pitchers_fastball():
    out = features.transform(pd.DataFrame([looked_up_row()]))
    assert out["pitch_velocity_diff_from_fastball"].iloc[0] == pytest.approx(86 - 96)
    assert out["pitch_horizontal_break_diff_from_fastball"].iloc[0] == pytest.approx(-4 - 15)


def test_cell_features_pick_the_pitches_own_cell():
    df = pd.DataFrame([looked_up_row(), looked_up_row(is_in_zone=True)])
    out = features.transform(df)
    assert out["cell_swing_rate"].tolist() == pytest.approx([0.37, 0.9])
    assert out["cell_pitches"].iloc[0] == 120


def test_missing_lookups_and_other_pitch_groups_give_nan():
    out = features.transform(
        pd.DataFrame([looked_up_row(pitch_group="other", velocity=None, pitch_type="EP")])
    )
    assert np.isnan(out["cell_swing_rate"].iloc[0])
    assert np.isnan(out["pitch_velocity_diff_from_fastball"].iloc[0])


def test_unknown_categories_become_missing_not_new_levels():
    out = features.transform(pd.DataFrame([looked_up_row(pitch_type="ZZ")]))
    assert pd.isna(out["pitch_type"].iloc[0])
    assert list(out["pitch_type"].cat.categories) == features.CATEGORIES["pitch_type"]


def test_game_state_is_derived_from_the_request_columns():
    df = pd.DataFrame(
        [
            looked_up_row(),  # 1-2, runner on first, third inning, tied
            looked_up_row(balls=0, strikes=0, base_state="_2_", inning=8, bat_score_diff=-3),
            looked_up_row(balls=3, strikes=1, base_state="___", inning=9, bat_score_diff=-3),
            looked_up_row(batter_stands="L", base_state=None),
        ]
    )
    out = features.transform(df)
    assert out["count_group"].tolist() == ["two_strikes", "neutral", "hitter_ahead", "two_strikes"]
    assert out["is_first_pitch_of_pa"].tolist() == [0, 1, 0, 0]
    assert out["runners_on"].tolist()[:3] == [1, 1, 0]
    assert out["runner_in_scoring_position"].tolist()[:3] == [0, 1, 0]
    # Down 3 in the 8th with a runner on: the runner, batter and on-deck hitter tie it
    assert out["is_late_and_close"].tolist()[:3] == [0, 1, 0]
    assert out["is_platoon_same_side"].tolist() == [1, 1, 1, 0]
    assert out.loc[3, ["runners_on", "is_late_and_close"]].isna().all()


def test_count_group_matches_the_batter_profile_cells():
    """transform() must put each count in the same group as the batter profile's SQL."""
    counts = [(b, s) for b in range(4) for s in range(3)]
    out = features.transform(pd.DataFrame([looked_up_row(balls=b, strikes=s) for b, s in counts]))
    for (balls, strikes), group in zip(counts, out["count_group"], strict=True):
        condition = batter_profile.COUNT_GROUPS[group]
        python = condition.replace("AND", "and").replace(" = ", " == ")
        assert eval(python, {"balls": balls, "strikes": strikes}), (balls, strikes, group)


def test_a_hypothetical_pitch_needs_no_pitch_key():
    """The recommender's rows: request columns and lookups only, no game_pk etc."""
    row = looked_up_row()
    hypothetical = pd.DataFrame([{k: row[k] for k in features.TRANSFORM_INPUTS}])
    assert not {"game_pk", "at_bat_number", "pitch_number"} & set(hypothetical.columns)
    assert list(features.transform(hypothetical).columns) == features.FEATURES


def test_labels_sql_has_the_request_columns_and_only_the_split_key():
    sql = features.labels_sql("g.fact_pitch", "TRUE", "int(is_swing)")
    for name in [*features.PITCH_COLUMNS, *features.STATE_COLUMNS, *features.LOOKUP_KEY_COLUMNS]:
        assert f"{name}" in sql
    assert "game_pk" in sql and "at_bat_number" not in sql and "season" not in sql


def test_should_promote():
    assert should_promote("always", 0.5, 2026, 0.4, 2026)[0]
    assert should_promote("if_better", 0.5, 2026, None, None)[0]
    assert should_promote("if_better", 0.4, 2026, 0.5, 2026)[0]
    assert not should_promote("if_better", 0.5, 2026, 0.4, 2026)[0]
    assert not should_promote("if_better", 0.4, 2026, 0.5, 2025)[0]  # different test season


def test_feature_lookups_never_return_model_label_or_key_columns():
    looked_up = set(features.BATTER_FEATURES + features.ARSENAL_FEATURES)
    assert not looked_up & {"label", *features.LOOKUP_KEY_COLUMNS}
    assert not looked_up & {*features.PITCH_COLUMNS, *features.STATE_COLUMNS}


def test_models_cover_the_pitch_result_tree():
    assert [m.name for m in MODELS] == [
        "swing", "take_outcome", "whiff", "contact_outcome", "in_play_outcome"
    ]  # fmt: skip
    assert BY_NAME["swing"].is_binary and not BY_NAME["in_play_outcome"].is_binary


def test_training_where_combines_filters():
    where = training_where(BY_NAME["whiff"], [2023, 2026], 10)
    assert "is_swing" in where and "season IN (2023, 2026)" in where
    assert "pmod(hash(game_pk), 100) < 10" in where
    assert "pmod" not in training_where(BY_NAME["whiff"], [2023], 100)


def test_parse_args_defaults():
    args = parse_args(
        ["--model", "swing", "--catalog", "c", "--gold-schema", "g", "--ml-schema", "m",
         "--experiment", "/x"]
    )  # fmt: skip
    assert args.train_seasons == [2023, 2024, 2025] and args.test_season == 2026


def test_log_loss_and_brier():
    y = np.array([0, 1])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    coin = np.full((2, 2), 0.5)
    assert evaluation.log_loss(y, perfect) == pytest.approx(0, abs=1e-12)
    assert evaluation.log_loss(y, coin) == pytest.approx(np.log(2))
    assert evaluation.brier(y, coin) == pytest.approx(0.5)


def test_frequency_baseline_smooths_and_falls_back():
    train = pd.DataFrame({"g": ["a", "a", "a", "a", "b"]})
    y = np.array([1, 1, 1, 1, 0])
    test = pd.DataFrame({"g": ["a", "z"]})
    probs = evaluation.frequency_baseline(train, y, test, 2, prior_weight=1.0)
    overall = np.array([0.2, 0.8])
    assert probs[0] == pytest.approx((np.array([0, 4]) + overall) / 5)
    assert probs[1] == pytest.approx(overall)  # unseen group
    prior = evaluation.frequency_baseline(train[[]], y, test[[]], 2)
    assert prior[0] == pytest.approx(overall)


def test_frequency_baseline_with_two_keys():
    train = pd.DataFrame({"b": [0, 0, 3], "s": [0, 0, 2]})
    y = np.array([1, 1, 0])
    test = pd.DataFrame({"b": [3, 0], "s": [2, 0]})
    probs = evaluation.frequency_baseline(train, y, test, 2, prior_weight=0.0)
    assert probs.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_calibration_table_and_ece():
    p = np.linspace(0.05, 0.95, 100)
    y = (np.random.default_rng(0).random(100) < p).astype(float)
    table = evaluation.calibration_table(y, p, bins=5)
    assert table["rows"].sum() == 100 and len(table) == 5
    perfect = pd.DataFrame(
        {"rows": [1, 1], "mean_predicted": [0.2, 0.8], "observed_rate": [0.2, 0.8]}
    )
    assert evaluation.expected_calibration_error(perfect) == 0


def test_fit_sample_condition():
    from databucs.jobs.train_outcome_model import fit_sample_condition

    assert fit_sample_condition(1_000, 1_500, 2026) == "TRUE"
    condition = fit_sample_condition(2_000_000, 1_500_000, 2026)
    assert "year(game_date) = 2026" in condition  # test rows always kept
    assert "game_pk % 10 = 0" in condition  # holdout rows always kept
    assert "< 750)" in condition  # 75% of fitting games


def test_transform_inputs_cover_everything_transform_reads():
    row = looked_up_row()
    trimmed = pd.DataFrame([{k: row[k] for k in features.TRANSFORM_INPUTS}])
    assert list(features.transform(trimmed).columns) == features.FEATURES


def test_result_probabilities_cover_every_result_once():
    from databucs.models.run_values import (
        COUNT_RESULTS,
        PA_ENDING_RESULTS,
        RESULT_PROBABILITY_SQL,
        RESULTS,
    )

    assert list(RESULT_PROBABILITY_SQL) == RESULTS
    assert sorted(COUNT_RESULTS + PA_ENDING_RESULTS) == sorted(RESULTS)


def test_result_probabilities_sum_to_one():
    """Evaluate the probability expressions with made-up model outputs."""
    from types import SimpleNamespace

    from databucs.models.run_values import RESULT_PROBABILITY_SQL

    def struct(**probs):
        return SimpleNamespace(**{f"p_{k}": v for k, v in probs.items()})

    p = SimpleNamespace(
        swing=struct(take=0.55, swing=0.45),
        take_outcome=struct(ball=0.6, called_strike=0.39, hit_by_pitch=0.01),
        whiff=struct(contact=0.75, whiff=0.25),
        contact_outcome=struct(foul=0.5, in_play=0.5),
        in_play_outcome=struct(out=0.68, single=0.2, double=0.07, triple=0.01, home_run=0.04),
    )
    total = sum(eval(expr, {"p": p}) for expr in RESULT_PROBABILITY_SQL.values())
    assert total == pytest.approx(1.0)


def test_prediction_struct_matches_the_pyfunc_columns():
    from databucs.models.run_values import prediction_struct_type

    assert prediction_struct_type("swing") == "struct<p_take: double, p_swing: double>"


def test_backtest_sql_groups_and_filters():
    from databucs.models import backtest

    sql = backtest.backtest_sql("m.preds", "g.fact_pitch")
    for name, keys in backtest.GROUPINGS.items():
        assert f"'{name}'" in sql
        assert f"PARTITION BY {', '.join(keys)}" in sql
    assert f"n >= {backtest.MIN_PITCHES}" in sql and "run_value IS NOT NULL" in sql


def test_backtest_summary_mean_and_standard_error():
    from databucs.models import backtest

    folds = pd.DataFrame(
        {
            "grouping": ["situation"] * 4,
            "fold": [0, 1, 2, 3],
            "pitches": [10, 10, 10, 10],
            "slope": [0.9, 1.1, 1.0, 1.0],
            "worst_minus_best_actual": [6.0, 8.0, 7.0, 7.0],
            "worst_minus_best_expected": [6.5] * 4,
        }
    )
    [row] = backtest.summarize(folds)
    assert row["pitches"] == 40 and row["slope"] == pytest.approx(1.0)
    assert row["slope_se"] == pytest.approx(folds["slope"].std() / 2)


def test_result_probabilities_in_python_match_the_sql_paths():
    from databucs.models.run_values import RESULTS, result_probabilities

    predictions = {
        "swing": {"p_take": 0.55, "p_swing": 0.45},
        "take_outcome": {"p_ball": 0.6, "p_called_strike": 0.39, "p_hit_by_pitch": 0.01},
        "whiff": {"p_contact": 0.75, "p_whiff": 0.25},
        "contact_outcome": {"p_foul": 0.5, "p_in_play": 0.5},
        "in_play_outcome": {
            "p_out": 0.68,
            "p_single": 0.2,
            "p_double": 0.07,
            "p_triple": 0.01,
            "p_home_run": 0.04,
        },
    }
    probs = result_probabilities(predictions)
    assert list(probs) == RESULTS
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["foul"] == pytest.approx(0.45 * 0.75 * 0.5)
