"""Check which external data sources this compute can reach.

Databricks Free Edition restricts outbound internet from serverless compute to a set of
trusted domains. Run this as a job to see which of the project's data sources are reachable
from serverless, or locally (`uv run python src/databucs/jobs/check_network.py`) to compare.

Informational only: it reports results and always exits successfully.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

USER_AGENT = "Mozilla/5.0 (compatible; databucs-network-check)"

# (name, url): a small request against each source the ingestion would use.
TARGETS: list[tuple[str, str]] = [
    ("control: example.com", "https://example.com/"),
    ("MLB Stats API", "https://statsapi.mlb.com/api/v1/sports"),
    ("Baseball Savant home", "https://baseballsavant.mlb.com/"),
    (
        "Baseball Savant CSV search",
        "https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details"
        "&player_type=pitcher&game_date_gt=2025-09-01&game_date_lt=2025-09-01",
    ),
    (
        "GitHub raw (Chadwick player IDs)",
        "https://raw.githubusercontent.com/chadwickbureau/register/master/README.md",
    ),
    ("PyPI (pybaseball)", "https://pypi.org/simple/pybaseball/"),
]


@dataclass
class CheckResult:
    name: str
    url: str
    ok: bool
    detail: str


def check_url(
    name: str,
    url: str,
    timeout: float = 15.0,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> CheckResult:
    """Request `url` and read a little of the body. Never raises."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener(request, timeout=timeout) as response:  # type: ignore[attr-defined]
            first_bytes = response.read(200)
            return CheckResult(name, url, True, f"HTTP {response.status}, {len(first_bytes)}B read")
    except urllib.error.HTTPError as e:
        # The server (or a proxy in front of it) answered, but with an error status.
        return CheckResult(name, url, False, f"HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        return CheckResult(name, url, False, f"connection failed: {e.reason}")
    except Exception as e:  # timeouts, TLS errors, etc.
        return CheckResult(name, url, False, f"{type(e).__name__}: {e}")


def format_report(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = [f"{'OK' if r.ok else 'FAIL':<4}  {r.name:<{width}}  {r.detail}" for r in results]
    reachable = sum(r.ok for r in results)
    lines.append(f"\n{reachable}/{len(results)} reachable")
    return "\n".join(lines)


def main() -> None:
    results = [check_url(name, url) for name, url in TARGETS]
    print(format_report(results))


if __name__ == "__main__":
    main()
