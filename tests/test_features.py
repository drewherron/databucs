import argparse
import datetime as dt

import pytest

from databucs.features import batter_profile, game_state, pitcher_arsenal
from databucs.features.common import WINDOW_DAYS, Source, date_filter, shrunk, trailing_window
from databucs.jobs.build_features import parse_args, parse_bool, select_tables
from databucs.jobs.check_features import coverage_failures

SOURCE = Source("g.fact_pitch", "s.players", dt.date(2025, 6, 10), dt.date(2025, 6, 12))


def test_history_starts_a_full_window_before_the_first_day():
    assert SOURCE.history_start() == dt.date(2025, 6, 10) - dt.timedelta(days=WINDOW_DAYS - 1)
    assert SOURCE.all_days().history_start() is None


def test_date_filter():
    assert date_filter("d", None, None) == "TRUE"
    assert date_filter("d", dt.date(2025, 1, 2), None) == "d >= DATE'2025-01-02'"
    assert (
        date_filter("d", dt.date(2025, 1, 2), dt.date(2025, 1, 3))
        == "d >= DATE'2025-01-02' AND d <= DATE'2025-01-03'"
    )


def test_trailing_window_includes_current_day_and_previous_days():
    assert trailing_window("b") == (
        f"(PARTITION BY b ORDER BY unix_date(game_date) "
        f"RANGE BETWEEN {WINDOW_DAYS - 1} PRECEDING AND CURRENT ROW)"
    )


def test_shrunk_rate_moves_toward_league():
    # Evaluate the SQL expression as Python arithmetic
    expr = shrunk("s", "n", "lg", 40)
    assert eval(expr, {"s": 0, "n": 0, "lg": 0.3}) == pytest.approx(0.3)  # no data: league
    assert eval(expr, {"s": 8, "n": 10, "lg": 0.3}) == pytest.approx((8 + 12) / 50)
    assert eval(expr, {"s": 800, "n": 1000, "lg": 0.3}) == pytest.approx(812 / 1040)


def test_batter_cells_cover_each_group_region_and_count_once():
    names = [name for name, _ in batter_profile.cells()]
    assert len(names) == len(set(names)) == 3 * 2 * 3
    assert "breaking_chase_two_strikes" in names


def test_count_groups_partition_every_count():
    groups = batter_profile.COUNT_GROUPS
    for balls in range(4):
        for strikes in range(3):
            matches = [
                name
                for name, cond in groups.items()
                if eval(
                    cond.replace("AND", "and").replace(" = ", " == "),
                    {"balls": balls, "strikes": strikes},
                )
            ]
            assert len(matches) == 1, (balls, strikes, matches)


@pytest.mark.parametrize("module", [batter_profile, pitcher_arsenal, game_state])
def test_sql_reads_the_window_and_returns_the_keys(module):
    sql = module.build_sql(SOURCE)
    assert "g.fact_pitch" in sql
    assert "DATE'2025-06-12'" in sql
    for key in module.PRIMARY_KEYS:
        assert key in sql
    if module.TIMESERIES_COLUMN:
        assert "date_add(" in sql  # rows become usable the day after their last game day
        assert "DATE'2025-06-10'" in sql  # output limited to the window


def test_unbounded_sql_has_no_date_limits():
    for module in (batter_profile, pitcher_arsenal, game_state):
        assert "DATE'" not in module.build_sql(SOURCE.all_days())


def test_parse_bool():
    assert parse_bool("") is False
    assert parse_bool("False") is False
    assert parse_bool("true") is True
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("maybe")


def test_full_rebuild_flag_forms():
    base = ["--catalog", "c", "--silver-schema", "s", "--gold-schema", "g", "--ml-schema", "m"]
    assert parse_args(base).full_rebuild is False
    assert parse_args([*base, "--full-rebuild"]).full_rebuild is True
    assert parse_args([*base, "--full-rebuild=false"]).full_rebuild is False


def test_select_tables():
    assert len(select_tables(None)) == 3
    assert select_tables(["game_state_features"]) == [game_state]
    with pytest.raises(ValueError):
        select_tables(["nope"])


def test_coverage_failures():
    assert coverage_failures(100, 99, 98, 100) == []
    assert coverage_failures(0, 0, 0, 0) == ["no pitches in the sample"]
    assert len(coverage_failures(100, 50, 99, 99)) == 2
