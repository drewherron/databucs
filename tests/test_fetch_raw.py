import datetime as dt
import json

from databucs.jobs.fetch_raw import parse_args, run, summarize

D = dt.date

SCHEDULE = json.dumps(
    {
        "dates": [
            {"date": "2025-07-04", "games": [{"status": {"abstractGameState": "Final"}}]},
            {"date": "2025-07-05", "games": [{"status": {"abstractGameState": "Final"}}]},
        ]
    }
).encode()

CSV = b'"pitch_type","game_pk"\n"FF","1"\n"SL","1"\n'


def fake_fetch(statcast_by_date=None):
    statcast_by_date = statcast_by_date or {}

    def fetch(url):
        if "/schedule?" in url:
            return SCHEDULE
        if "statcast_search" in url:
            for day, body in statcast_by_date.items():
                if f"game_date_gt={day}" in url:
                    return body
            return CSV
        return b'{"ok": true}'  # teams, players

    return fetch


def landed_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_run_lands_schedule_statcast_teams_and_players(tmp_path):
    outcomes = run(tmp_path, D(2025, 7, 4), D(2025, 7, 5), fetch=fake_fetch())
    files = landed_files(tmp_path)
    assert [f.split("/")[0:2] for f in files] == [
        ["statcast", "2025"],
        ["statcast", "2025"],
        ["statsapi", "players"],
        ["statsapi", "schedule"],
        ["statsapi", "teams"],
    ]
    assert all(o.written and not o.error for o in outcomes)


def test_rerun_with_same_data_writes_nothing(tmp_path):
    run(tmp_path, D(2025, 7, 4), D(2025, 7, 5), fetch=fake_fetch())
    before = landed_files(tmp_path)
    outcomes = run(tmp_path, D(2025, 7, 4), D(2025, 7, 5), fetch=fake_fetch())
    assert landed_files(tmp_path) == before
    assert not any(o.written for o in outcomes)


def test_errors_and_empty_days_are_reported_not_raised(tmp_path):
    fetch = fake_fetch({"2025-07-04": b"<html>oops</html>", "2025-07-05": b""})
    outcomes = run(tmp_path, D(2025, 7, 4), D(2025, 7, 5), fetch=fetch)
    errors = {o.item: o.error for o in outcomes if o.error}
    assert errors["statcast 2025-07-04"].startswith("SavantResponseError")
    assert errors["statcast 2025-07-05"] == "no pitches returned"
    assert "2 errors" in summarize(outcomes)


def test_parse_args_treats_empty_dates_as_unset():
    args = parse_args(["--landing-path", "/tmp/x", "--start-date=", "--end-date=2025-07-05"])
    assert args.start_date is None
    assert args.end_date == D(2025, 7, 5)


def test_old_days_already_landed_are_skipped_but_recent_days_refetched(tmp_path):
    run(tmp_path, D(2025, 7, 4), D(2025, 7, 5), fetch=fake_fetch())
    changed = fake_fetch({"2025-07-04": CSV + b'"CH","1"\n', "2025-07-05": CSV + b'"CU","1"\n'})
    outcomes = run(
        tmp_path, D(2025, 7, 4), D(2025, 7, 5), fetch=changed, refetch_before=D(2025, 7, 5)
    )
    by_item = {o.item: o for o in outcomes}
    assert by_item["statcast 2025-07-04"].skipped  # old day: not re-downloaded
    assert by_item["statcast 2025-07-05"].written  # recent day: correction picked up
    assert "1 skipped" in summarize(outcomes)


def test_parse_args_refetch_all_defaults_off():
    assert not parse_args(["--landing-path", "/tmp/x"]).refetch_all
