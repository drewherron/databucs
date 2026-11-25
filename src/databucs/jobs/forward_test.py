"""Forward-in-time test of the simulator (databucs.models.forward_test).

1. Pick the pitchers (busiest after --as-of, with enough plate appearances before it too)
   and a panel of hitters (most plate appearances before it).
2. Score them as of that date with recommend_pitches (the simulator's scenarios only) and
   simulate with simulate_matchups, into their own tables so the dashboard isn't touched:
   forward_test_option_scores and forward_test_simulations. Team and opponent ids are 0.
3. Compare each pitcher's simulated K% and BB% (usual mix, averaged over the panel) with
   what he did after the date, next to the naive forecast (his rate before it), and log
   the summary to the outcome_models MLflow experiment.

--model-alias picks the outcome model versions, so a challenger can take the same test.
"""

from __future__ import annotations

import argparse

from databucs.ingestion.dates import optional_date
from databucs.jobs import recommend_pitches, simulate_matchups
from databucs.models import forward_test
from databucs.models.scoring import CHAMPION_ALIAS

OPTIONS_TABLE = "forward_test_option_scores"
SIMULATIONS_TABLE = "forward_test_simulations"
TEST_ID = 0  # team and opponent id of the test's rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-in-time test of the simulator.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--as-of", type=optional_date, default=None, help="Default 2026-07-01.")
    parser.add_argument("--end", type=optional_date, default=None, help="Default: season end.")
    parser.add_argument("--pitchers", type=int, default=60)
    parser.add_argument("--min-pa", type=int, default=150, help="On each side of the date.")
    parser.add_argument("--batters", type=int, default=9)
    parser.add_argument("--model-alias", default=CHAMPION_ALIAS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import datetime as dt

    args = parse_args(argv)
    as_of = args.as_of or dt.date(2026, 7, 1)
    end = args.end or dt.date(as_of.year, 12, 31)
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"
    ml = f"{args.catalog}.{args.ml_schema}"

    import mlflow
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    # 1. Pitchers and hitters
    outcomes = spark.sql(forward_test.pitcher_outcomes_sql(fact_pitch, str(as_of), str(end)))
    outcomes = outcomes.toPandas()
    outcomes.attrs = {}
    pitchers = forward_test.choose_pitchers(outcomes, args.pitchers, args.min_pa)
    batters = [
        r.batter_id
        for r in spark.sql(
            forward_test.batter_panel_sql(fact_pitch, str(as_of), args.batters)
        ).collect()
    ]
    print(f"As of {as_of}: {len(pitchers)} pitchers, {len(batters)} hitters")

    # 2. Score and simulate as of the date
    common = [
        f"--catalog={args.catalog}",
        f"--gold-schema={args.gold_schema}",
        f"--ml-schema={args.ml_schema}",
        f"--team-id={TEST_ID}",
        f"--opponent-id={TEST_ID}",
        f"--series-start={as_of}",
    ]
    recommend_pitches.main(
        [
            *common,
            f"--silver-schema={args.silver_schema}",
            f"--lineup={','.join(map(str, batters))}",
            f"--pitchers={','.join(map(str, pitchers))}",
            f"--table={OPTIONS_TABLE}",
            f"--model-alias={args.model_alias}",
            "--simulation-only",
        ]
    )
    simulate_matchups.main(
        [*common, f"--options-table={OPTIONS_TABLE}", f"--table={SIMULATIONS_TABLE}"]
    )

    # 3. Compare with what happened after the date
    simulated = spark.sql(
        f"""
        SELECT pitcher_id, any_value(pitcher_name) AS pitcher_name,
          avg(p_strikeout) AS sim_k, avg(p_walk) AS sim_bb, avg(run_value) AS sim_run_value,
          count(*) AS batters
        FROM {ml}.{SIMULATIONS_TABLE}
        WHERE team_id = {TEST_ID} AND series_start = DATE'{as_of}' AND policy = 'usual'
        GROUP BY pitcher_id
        """
    ).toPandas()
    simulated.attrs = {}
    df = simulated.merge(outcomes, on="pitcher_id")
    summary = forward_test.summarize(df)
    print(summary.round(4).to_string(index=False))

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="simulator_forward_test"):
        mlflow.log_params(
            {
                "as_of": str(as_of),
                "end": str(end),
                "pitchers": len(df),
                "batters": len(batters),
                "min_pa": args.min_pa,
                "model_alias": args.model_alias,
            }
        )
        for row in summary.to_dict("records"):
            for key, value in row.items():
                if key not in ("rate", "pitchers"):
                    mlflow.log_metric(f"{row['rate']}_{key}", float(value))
        mlflow.log_table(summary, "forward_test_summary.json")
        mlflow.log_table(df, "forward_test_pitchers.json")


if __name__ == "__main__":
    main()
