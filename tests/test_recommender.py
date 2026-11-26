import datetime as dt

import pytest

from databucs.jobs.recommend_pitches import parse_args
from databucs.models import features
from databucs.recommender import inputs, options
from databucs.recommender.inputs import Game, choose_series

PIT, DET, STL = 134, 116, 138


def game(pk, day, home, away, played, number, home_sp=None, away_sp=None):
    return Game(pk, dt.date(2026, 9, day), home, away, played, number, home_sp, away_sp)


GAMES = [
    game(1, 22, PIT, STL, True, 1),
    game(2, 23, PIT, STL, True, 2),
    game(3, 24, PIT, STL, True, 3),
    game(4, 25, DET, PIT, False, 1, 901, 701),
    game(5, 26, DET, PIT, False, 2, 902, 702),
    game(6, 27, DET, PIT, False, 3, 903, 701),
]


def test_next_series_within_the_horizon():
    target = choose_series(GAMES, PIT, dt.date(2026, 9, 24), 10)
    assert (target.mode, target.opponent_id, target.series_start) == (
        "scheduled", DET, dt.date(2026, 9, 25)
    )  # fmt: skip
    assert target.game_pks == [4, 5, 6]
    assert target.probable_starters == [701, 702]  # the Pirates' side, without repeats


def test_demo_on_the_last_series_played_when_nothing_is_close():
    target = choose_series(GAMES, PIT, dt.date(2026, 9, 1), 10)
    assert (target.mode, target.opponent_id, target.series_start) == (
        "demo", STL, dt.date(2026, 9, 22)
    )  # fmt: skip
    assert target.game_pks == [1, 2, 3]


def test_chosen_and_custom_series():
    chosen = choose_series(GAMES, PIT, dt.date(2026, 9, 1), 10, dt.date(2026, 9, 25))
    assert (chosen.mode, chosen.opponent_id) == ("chosen", DET)
    custom = choose_series(GAMES, PIT, dt.date(2026, 9, 1), 10, dt.date(2027, 4, 1), DET)
    assert (custom.mode, custom.game_pks) == ("custom", [])
    with pytest.raises(ValueError):
        choose_series(GAMES, PIT, dt.date(2026, 9, 1), 10, dt.date(2027, 4, 1))


def test_hypothetical_pitch_supplies_every_pitch_column():
    assert list(inputs.HYPOTHETICAL_PITCH) == list(features.PITCH_COLUMNS)


def test_each_zone_has_nine_distinct_points():
    points = options.zone_points()
    assert len(points) == 9 * len(options.ZONES)
    for zone in options.ZONES:
        coords = {(p.plate_x, p.height) for p in points if p.zone == zone}
        assert len(coords) == 9


def in_strike_zone(p):
    return abs(p.plate_x) <= options.PLATE_HALF_WIDTH and 0 <= p.height <= 1


def test_in_zone_points_fall_in_their_own_cell():
    third = 2 * options.PLATE_HALF_WIDTH / 3
    for p in options.zone_points():
        if p.zone in options.IN_ZONE:
            column, row = (p.zone - 1) % 3, (p.zone - 1) // 3
            assert in_strike_zone(p)
            assert int((p.plate_x + options.PLATE_HALF_WIDTH) // third) == column
            assert int((1 - p.height) * 3) == row


def test_chase_points_are_outside_the_zone_in_their_quadrant():
    for p in options.zone_points():
        if p.zone in options.CHASE_ZONES:
            side, end = options.CHASE_ZONES[p.zone]
            assert not in_strike_zone(p)
            assert p.plate_x * side > 0
            assert (p.height > 0.5) == (end == "high")
            # near the zone, not in the dirt or over the batter's head
            assert abs(p.plate_x) <= options.PLATE_HALF_WIDTH + 0.5
            assert -0.3 <= p.height <= 1.3


def test_scenarios_are_valid_states():
    names = [s.name for s in options.SCENARIOS]
    assert len(names) == len(set(names))
    for s in options.SCENARIOS:
        assert 0 <= s.balls <= 3 and 0 <= s.strikes <= 2 and 0 <= s.outs <= 2
        assert len(s.base_state) == 3


def test_scenarios_cover_every_count_where_the_simulator_starts():
    counts = {
        (s.balls, s.strikes)
        for s in options.SCENARIOS
        if (s.outs, s.base_state) == (options.SIMULATION_OUTS, options.SIMULATION_BASE_STATE)
    }
    assert counts == {(b, s) for b in range(4) for s in range(3)}


def test_arsenal_keeps_established_pitches_and_counts_all_of_them():
    target = inputs.Target(PIT, DET, dt.date(2026, 9, 25), "demo")
    sql = inputs.arsenal_sql("m.arsenal", "g.fact_pitch", target)
    assert f"a.pitches_thrown >= {inputs.ESTABLISHED_PITCHES}" in sql
    assert f"a.usage_rate >= {inputs.ESTABLISHED_USAGE_RATE}" in sql
    # The total is taken before the filter, in the CTE
    assert sql.index("pitcher_total_pitches") < sql.index("WHERE a.pitches_thrown")


def test_count_group_sql_names_every_group():
    sql = inputs.count_group_sql("o.balls", "o.strikes")
    assert "o.balls > o.strikes" in sql and "o.strikes = 2" in sql
    for name in ("two_strikes", "hitter_ahead", "neutral"):
        assert f"'{name}'" in sql


def test_request_rows_use_only_data_before_the_series():
    target = inputs.Target(PIT, DET, dt.date(2026, 9, 25), "demo")
    sql = inputs.request_rows_sql("g.fact_pitch", "s.players", "m.arsenal", target)
    assert "game_date < DATE'2026-09-25'" in sql
    assert "as_of_date <= DATE'2026-09-25'" in sql
    for name in [*features.PITCH_COLUMNS, *features.STATE_COLUMNS, *features.LOOKUP_KEY_COLUMNS]:
        assert name in sql


def test_parse_args_with_unset_job_parameters():
    args = parse_args(
        ["--catalog", "c", "--silver-schema", "s", "--gold-schema", "g", "--ml-schema", "m",
         "--opponent-id=", "--series-start=", "--lineup=", "--pitchers="]
    )  # fmt: skip
    assert args.opponent_id is None and args.series_start is None
    assert args.lineup == [] and args.pitchers == [] and args.team_id == PIT
    custom = parse_args(
        ["--catalog", "c", "--silver-schema", "s", "--gold-schema", "g", "--ml-schema", "m",
         "--lineup=1,2,3"]
    )  # fmt: skip
    assert custom.lineup == [1, 2, 3]


def test_statcast_zone_matches_the_aim_points_zones():
    for p in options.zone_points():
        assert options.statcast_zone(p.plate_x, p.height) == p.zone


def test_landing_grid_covers_every_zone_and_indexes_in_order():
    points = options.landing_points()
    assert [p.land for p in points] == list(range(len(points)))
    assert {p.zone for p in points} == set(options.ZONES)


def test_landing_index_sql_picks_the_nearest_point():
    """Evaluate the SQL expression as Python, for a few locations."""
    sql = options.landing_index_sql("x", "h")
    python = sql.replace("least", "min").replace("greatest", "max")
    points = options.landing_points()
    for x, h in [(0.0, 0.5), (-0.9, 1.1), (5.0, -3.0), (0.61, 0.07)]:
        land = eval(python, {"x": x, "h": h, "min": min, "max": max, "int": int, "round": round})
        nearest = min(points, key=lambda p: (abs(p.plate_x - x), abs(p.height - h)))
        clipped = min(
            points,
            key=lambda p: (
                abs(p.plate_x - max(min(x, 1.5), -1.5)),
                abs(p.height - max(min(h, 1.5), -0.5)),
            ),
        )
        assert land in (nearest.land, clipped.land)


def test_aim_weights_sum_to_one_and_center_on_the_aim():
    weights = options.aim_weights(zone_height_ft=1.9, sd_ft=0.65)
    landing = {p.land: p for p in options.landing_points()}
    by_aim: dict = {}
    for zone, point, land, w in weights:
        by_aim.setdefault((zone, point), []).append((land, w))
    assert len(by_aim) == 9 * len(options.ZONES)
    for cells in by_aim.values():
        assert sum(w for _, w in cells) == pytest.approx(1)
    # Aimed at the middle: about 68% land in the zone (see MISS_SD_FT)
    middle = by_aim[(5, 4)]
    in_zone = sum(w for land, w in middle if landing[land].zone in options.IN_ZONE)
    assert 0.6 < in_zone < 0.75
    # A pitcher who never misses lands at the aim point (to within the grid's spacing)
    aims = {(a.zone, a.point): a for a in options.zone_points()}
    sharp: dict = {}
    for zone, point, land, w in options.aim_weights(zone_height_ft=1.9, sd_ft=0.001):
        sharp.setdefault((zone, point), []).append((landing[land], w))
    for key, cells in sharp.items():
        x = sum(p.plate_x * w for p, w in cells)
        h = sum(p.height * w for p, w in cells)
        assert abs(x - aims[key].plate_x) <= options.LANDING_DX / 2 + 1e-9
        assert abs(h - aims[key].height) <= options.LANDING_DH / 2 + 1e-9


def test_wider_misses_spread_further():
    def in_zone_share(sd):
        landing = {p.land: p for p in options.landing_points()}
        return sum(
            w
            for zone, point, land, w in options.aim_weights(1.9, sd)
            if (zone, point) == (5, 4) and landing[land].zone in options.IN_ZONE
        )

    assert in_zone_share(0.4) > in_zone_share(0.65) > in_zone_share(0.9)


def test_zone_labels_are_from_the_batters_point_of_view():
    # Zone 1 is the catcher's upper left: up and in to a righty, up and away to a lefty
    assert options.zone_label(1, "R").location == "up and in"
    assert options.zone_label(1, "L").location == "up and away"
    assert options.zone_label(14, "L").location == "chase low and in"
    assert options.zone_label(14, "R").location == "chase low and away"
    assert options.zone_label(5, "R").location == "middle"
    assert options.zone_label(8, "L").location == "low, over the middle"
    labels = options.zone_labels()
    for side in ("L", "R"):
        cells = {(z.grid_x, z.grid_y) for z in labels if z.batter_stands == side}
        assert len(cells) == len(options.ZONES)  # every zone has its own grid cell
