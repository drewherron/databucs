import urllib.error

from databucs.jobs.check_network import CheckResult, check_url, format_report


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        return b"x" * min(n, 50)


def raising(exc):
    def opener(request, timeout):
        raise exc

    return opener


def test_check_url_success():
    result = check_url(
        "site", "https://example.com", opener=lambda request, timeout: FakeResponse()
    )
    assert result.ok
    assert result.detail == "HTTP 200, 50B read"


def test_check_url_http_error():
    err = urllib.error.HTTPError("https://example.com", 403, "Forbidden", {}, None)
    result = check_url("site", "https://example.com", opener=raising(err))
    assert not result.ok
    assert result.detail == "HTTP 403 Forbidden"


def test_check_url_connection_error():
    result = check_url("site", "https://example.com", opener=raising(urllib.error.URLError("nope")))
    assert not result.ok
    assert "connection failed" in result.detail


def test_check_url_never_raises_on_unexpected_errors():
    result = check_url("site", "https://example.com", opener=raising(TimeoutError("slow")))
    assert not result.ok
    assert result.detail.startswith("TimeoutError")


def test_format_report_counts_reachable():
    report = format_report(
        [CheckResult("a", "u1", True, "HTTP 200"), CheckResult("bb", "u2", False, "HTTP 403")]
    )
    assert "OK    a   HTTP 200" in report
    assert "FAIL  bb  HTTP 403" in report
    assert report.endswith("1/2 reachable")
