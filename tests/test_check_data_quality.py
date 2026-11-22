import datetime as dt

from databucs.jobs.check_data_quality import Day, classify, format_day, parse_args

D1, D2 = dt.date(2025, 6, 1), dt.date(2025, 6, 2)


def day(game_date, *issues, played=15, with_pitches=15):
    return Day(game_date, played, with_pitches, 4400, 0, 0, tuple(issues))


def test_clean_days_have_no_failures_or_warnings():
    assert classify([day(D1), day(D2)], newest=D2) == ([], [])


def test_missing_pitches_on_newest_day_is_only_a_warning():
    failures, warnings = classify([day(D2, "games_missing_pitches", with_pitches=14)], newest=D2)
    assert failures == []
    assert len(warnings) == 1 and warnings[0].startswith("2025-06-02: games_missing_pitches")


def test_missing_pitches_on_older_day_fails():
    failures, _ = classify([day(D1, "games_missing_pitches")], newest=D2)
    assert failures == ["2025-06-01: games_missing_pitches"]


def test_other_issues_fail_even_on_newest_day():
    failures, warnings = classify([day(D2, "games_missing_pitches", "pitches_rejected")], newest=D2)
    assert failures == ["2025-06-02: pitches_rejected"]
    assert len(warnings) == 1


def test_format_day_lists_issues():
    line = format_day(day(D1, "game_with_few_pitches"))
    assert line.startswith("2025-06-01: 15/15 games with pitches, 4400 pitches")
    assert line.endswith("ISSUES: game_with_few_pitches")


def test_parse_args_treats_empty_dates_as_unset():
    args = parse_args(["--catalog", "c", "--silver-schema", "s", "--start-date", ""])
    assert args.start_date is None and args.end_date is None
