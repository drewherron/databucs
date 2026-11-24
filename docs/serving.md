# Real-time recommender (Model Serving)

The batch job scores a whole series ahead of time. The serving endpoint answers one
question at a time: *what should this pitcher throw this batter right now?* It covers any
count, outs and runners, inning and score, including situations the batch scenarios don't.

## How it's built

`register_recommender` job (`src/databucs/jobs/register_recommender.py`):

1. **Snapshot**, as of `as_of_date` (default today). The job takes everything the
   recommender looks up and saves it as parquet: batter strike zones and sides, pitcher
   hands, arsenals, the latest batter profiles, every pitcher's usual mix for every count,
   and the run values.
   - It's built with the batch job's own queries (`recommender/inputs.py`).
   - It covers everyone who batted or pitched in the last 365 days.
2. **Boosters:** the five `@champion` outcome models' LightGBM boosters.
3. **Consistency check:** when the batch table has a series starting on that date, the job
   recomputes 20 sampled situations with the pandas engine and fails if any option differs
   by more than 1e-6 runs or gets a different tier.
4. **Register:** it logs a pyfunc (`recommender/pyfunc.py`) with the snapshot and boosters
   as artifacts and registers it as `<ml schema>.pitch_recommender` with `@champion`.
5. **Serve:** it creates the endpoint `pitch-recommender-<target>` (CPU, Small, scales to
   zero) or moves it to the new version, then sends one request.

**Why a snapshot instead of online feature lookups?** Free Edition has no online tables,
and the recommender expands one request into every pitch type × landing spot inside the
model, while Feature Engineering lookups run on the request rows before it. A dated
snapshot keeps the model self-contained. To refresh it, rerun the job: daily in season, or
before a series.

**Same logic, two engines.** The batch job is Spark SQL and the endpoint is pandas
(`recommender/serving.py`). They share the geometry, tiers, labels and queries, and the
consistency check holds them together.
- On the Detroit demo (as of 2026-09-25), the largest difference over 20 situations was
  3e-17 runs.
- Skenes against Riley Greene at 0-2 returns the batch table's top options exactly.

## Asking it

Input: one row per question. Output: one row per option, ranked within each request.

| Input | |
|---|---|
| `pitcher_id`, `batter_id` | MLBAM ids |
| `balls`, `strikes`, `outs` | the count and outs |
| `base_state` | runners, e.g. `___`, `1__`, `_2_`, `1_3`, `123` |
| `inning`, `bat_score_diff`, `times_through_order` | optional (defaults 5, 0, 2) |

Params: `top_n` (default 10, 0 for all) and `miss_sd_ft` (default 0.65).

The output columns match `pitch_option_scores`: rank, tier, pitch, location, expected run
value and range, `vs_usual`, his share, the result probabilities, and the snapshot date.
An unknown player or invalid situation returns one row with `error` set, and the other
questions in the request are still answered.

```bash
databricks api post /serving-endpoints/pitch-recommender-dev/invocations --json '{
  "dataframe_records": [{"pitcher_id": 694973, "batter_id": 682985,
                         "balls": 0, "strikes": 2, "outs": 1, "base_state": "___"}],
  "params": {"top_n": 3}}'
```

`databricks serving-endpoints query` drops `params`, so use the invocations route above
when you need them.

## Known limitation: one miss size for everyone

Options assume a league-wide miss of 0.65 ft, while a pitcher's usual mix is valued at
where his pitches really landed. For a pitcher whose command is better than that, his
usual mix can beat every aimed option, and then every option shows as "avoid".
- The endpoint's example request (the snapshot's busiest pitcher at 1-2 with a runner on
  third) is one such case.
- In the Detroit batch table, the best option is worse than the usual mix in 0.5–7% of
  situations, mostly for a few pitchers (Antwone Kelly 16 of 112, Kirby Yates 14, Jared
  Jones 11).
- A per-pitcher miss size would fix it, but it needs intended targets, which public data
  doesn't have.

## Notes

- **Free Edition:** CPU custom-model endpoints are allowed, with a cap on how many.
  Scale-to-zero means the first request after idling waits for a cold start.
- **Spark Connect quirk:** `toPandas()` attaches query metrics to `DataFrame.attrs`, and
  pandas can't write those to parquet, so the job clears them.
- **Snapshot size:** the usual-mix tables are the largest part, with about 1.3M rows by
  landing spot and 0.5M by zone.
