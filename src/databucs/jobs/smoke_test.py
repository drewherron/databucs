"""Smoke test for a deployed bundle target.

Runs on Databricks serverless compute and checks that Spark works and that every
schema this bundle target expects exists in the catalog. Exits non-zero if not,
so the job run fails visibly.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument(
        "--schemas",
        required=True,
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        help="Comma-separated schema names expected in the catalog.",
    )
    return parser.parse_args(argv)


def missing_schemas(expected: list[str], existing: list[str]) -> list[str]:
    """Return the expected schemas that are not present (case-insensitive)."""
    existing_lower = {s.lower() for s in existing}
    return [s for s in expected if s.lower() not in existing_lower]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    from pyspark.sql import SparkSession  # imported here so unit tests don't need Spark

    spark = SparkSession.builder.getOrCreate()

    user, spark_version = spark.sql("SELECT current_user(), version()").first()
    print(f"Running as {user} on Spark {spark_version}")

    existing = [row[0] for row in spark.sql(f"SHOW SCHEMAS IN `{args.catalog}`").collect()]
    missing = missing_schemas(args.schemas, existing)
    if missing:
        print(f"Missing schemas in {args.catalog}: {', '.join(missing)}")
        return 1

    print(f"All schemas present in {args.catalog}: {', '.join(args.schemas)}")
    return 0


if __name__ == "__main__":
    # Databricks runs scripts in an IPython kernel, which treats any SystemExit (even 0) as a
    # failure, so only exit explicitly when something is wrong.
    if (code := main()) != 0:
        sys.exit(code)
