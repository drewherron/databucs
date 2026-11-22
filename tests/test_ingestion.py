import datetime as dt
import json
import urllib.error
import urllib.parse

import pytest

from databucs.ingestion import savant, statsapi
from databucs.ingestion.dates import resolve_window, split_by_month
from databucs.ingestion.http import FetchError, get_bytes
from databucs.ingestion.landing import content_hash, land

D = dt.date

# --- savant ---------------------------------------------------------------------------

STATCAST_CSV = '﻿"pitch_type","game_date","game_pk"\n"FF","2025-07-04","1"\n"SL","2025-07-04","1"\n'


def test_statcast_url_requests_one_day_of_regular_and_postseason_games():
    query = urllib.parse.parse_qs(
        urllib.parse.urlparse(savant.statcast_day_url(D(2025, 7, 4))).query
    )
    assert query["game_date_gt"] == query["game_date_lt"] == ["2025-07-04"]
    assert query["hfGT"] == ["R|F|D|L|W|"]


def test_count_rows_handles_byte_order_mark():
    assert savant.count_rows(STATCAST_CSV.encode()) == 2


def test_count_rows_empty_body_means_no_pitches():
    assert savant.count_rows(b"") == 0


def test_count_rows_rejects_non_csv():
    with pytest.raises(savant.SavantResponseError):
        savant.count_rows(b"<html>Service unavailable</html>")


def test_count_rows_rejects_truncated_response():
    body = "pitch_type\n" + "FF\n" * savant.ROW_CAP
    with pytest.raises(savant.SavantResponseError, match="truncated"):
        savant.count_rows(body.encode())


# --- statsapi -------------------------------------------------------------------------


def schedule(*days):
    """Build a minimal schedule response: days = [(date, [abstractGameState, ...]), ...]."""
    return json.dumps(
        {
            "dates": [
                {"date": d, "games": [{"status": {"abstractGameState": s}} for s in states]}
                for d, states in days
            ]
        }
    ).encode()


def test_completed_game_dates_skips_days_without_final_games():
    body = schedule(
        ("2025-07-05", ["Final", "Preview"]),
        ("2025-07-04", ["Final"]),
        ("2025-07-06", ["Preview"]),
    )
    assert statsapi.completed_game_dates(body) == [D(2025, 7, 4), D(2025, 7, 5)]


def test_completed_game_dates_skips_postponed_and_cancelled_games():
    body = json.dumps(
        {
            "dates": [
                {
                    "date": "2026-10-16",
                    "games": [{"status": {"abstractGameState": "Final", "codedGameState": "D"}}],
                },
                {
                    "date": "2026-10-17",
                    "games": [{"status": {"abstractGameState": "Final", "codedGameState": "C"}}],
                },
                {
                    "date": "2026-10-18",
                    "games": [{"status": {"abstractGameState": "Final", "codedGameState": "F"}}],
                },
            ]
        }
    ).encode()
    assert statsapi.completed_game_dates(body) == [D(2026, 10, 18)]


def test_completed_game_dates_uses_official_date_for_resumed_games():
    final = {"abstractGameState": "Final", "codedGameState": "F"}
    body = json.dumps(
        {
            "dates": [
                {
                    "date": "2023-10-02",
                    "games": [{"officialDate": "2023-09-28", "status": final}],
                },
            ]
        }
    ).encode()
    assert statsapi.completed_game_dates(body) == [D(2023, 9, 28)]


def test_schedule_url_includes_probables_and_lineups():
    query = urllib.parse.parse_qs(
        urllib.parse.urlparse(statsapi.schedule_url(D(2025, 3, 1), D(2025, 3, 31))).query
    )
    assert query["hydrate"] == ["probablePitcher,lineups"]
    assert query["gameType"] == ["R,F,D,L,W"]


# --- landing --------------------------------------------------------------------------


def test_land_writes_once_and_skips_identical_content(tmp_path):
    first = land(b"data", tmp_path / "x", "stem", "csv")
    second = land(b"data", tmp_path / "x", "stem", "csv")
    assert first.written and not second.written
    assert first.path == second.path == tmp_path / "x" / f"stem_{content_hash(b'data')}.csv"
    assert [p.name for p in (tmp_path / "x").iterdir()] == [first.path.name]  # no temp files


def test_land_keeps_changed_content_as_a_new_file(tmp_path):
    land(b"v1", tmp_path, "stem", "csv")
    assert land(b"v2", tmp_path, "stem", "csv").written
    assert len(list(tmp_path.iterdir())) == 2


# --- dates ----------------------------------------------------------------------------


def test_resolve_window_defaults_to_lookback_ending_yesterday():
    assert resolve_window(None, None, D(2025, 7, 10), lookback_days=3) == (
        D(2025, 7, 7),
        D(2025, 7, 9),
    )


def test_resolve_window_start_only_runs_through_yesterday():
    assert resolve_window(D(2025, 7, 1), None, D(2025, 7, 10)) == (D(2025, 7, 1), D(2025, 7, 9))


def test_resolve_window_rejects_reversed_range():
    with pytest.raises(ValueError):
        resolve_window(D(2025, 7, 9), D(2025, 7, 1), D(2025, 7, 10))


def test_split_by_month():
    assert split_by_month(D(2024, 12, 15), D(2025, 2, 10)) == [
        (D(2024, 12, 15), D(2024, 12, 31)),
        (D(2025, 1, 1), D(2025, 1, 31)),
        (D(2025, 2, 1), D(2025, 2, 10)),
    ]


def test_split_by_month_single_day():
    assert split_by_month(D(2025, 7, 4), D(2025, 7, 4)) == [(D(2025, 7, 4), D(2025, 7, 4))]


# --- http -----------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def flaky_opener(failures):
    """Raise each exception in `failures` in turn, then succeed."""
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return FakeResponse(b"ok")

    return opener, calls


def http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, None)


def test_get_bytes_retries_transient_errors():
    opener, calls = flaky_opener([http_error(503), urllib.error.URLError("reset")])
    assert get_bytes("https://example.com", opener=opener, sleep=lambda s: None) == b"ok"
    assert len(calls) == 3


def test_get_bytes_does_not_retry_client_errors():
    opener, calls = flaky_opener([http_error(404)])
    with pytest.raises(FetchError, match="404"):
        get_bytes("https://example.com", opener=opener, sleep=lambda s: None)
    assert len(calls) == 1


def test_get_bytes_gives_up_after_all_attempts():
    opener, _ = flaky_opener([http_error(503)] * 4)
    with pytest.raises(FetchError, match="4 attempts"):
        get_bytes("https://example.com", opener=opener, sleep=lambda s: None)
