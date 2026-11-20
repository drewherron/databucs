# databucs

An end-to-end data and ML platform on Databricks that produces advance-scouting reports: per-batter pitch recommendations and Pirates pitcher matchup rankings, built from public Statcast data.

> Personal project to learn Databricks. Not affiliated with the Pittsburgh Pirates or MLB.

## Status
- [x] Project setup: Asset Bundle (dev/prod), Unity Catalog schemas, CI
- [ ] Statcast and MLB Stats API ingestion

## Layout
```
databricks.yml        # Asset Bundle config (dev/prod targets)
resources/            # schemas, jobs (and later pipelines, serving, monitoring)
src/databucs/
tests/
.github/workflows/    # CI: lint + unit tests
```

## Setup
Requirements: [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install),
[uv](https://docs.astral.sh/uv/), a Databricks workspace (built on Free Edition).

```bash
uv sync                                   # local env (Python 3.12, matches serverless)
uv run pytest                             # unit tests
databricks auth login --host <workspace-url>
databricks bundle validate                # check config
databricks bundle deploy                  # deploy to dev (default target)
databricks bundle run smoke_test          # verify compute + schemas
```

Dev and prod live in the same workspace. Development mode prefixes schemas per user (`dev_<user>_bronze`), while prod uses plain names (`bronze`), all in the `workspace` catalog. Deploy prod with `databricks bundle deploy -t prod`.
