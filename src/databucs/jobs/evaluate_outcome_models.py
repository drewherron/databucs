"""Combine the five outcome models into expected run value and evaluate it on the test
season.

1. Run values per state and result from the training seasons (databucs.models.run_values).
2. Each @champion model scores the test season's competitive pitches with
   fe.score_batch, which looks up the batter and arsenal features itself; the pitch and
   game state columns come with each row.
3. The results' probabilities are multiplied along the outcome tree and weighted by their
   run values. The predictions are saved to <ml schema>.outcome_test_predictions.
4. Evaluation, logged to MLflow:
   - predicted vs actual share of each of the ten results
   - mean squared error of the pitches' actual run_value against expected run value,
     compared with a baseline that only knows the count and base-out state
   - actual run value by decile of expected run value (does the model rank pitches?)
"""

from __future__ import annotations

import argparse

from databucs.jobs.train_outcome_model import parse_seasons
from databucs.models import features
from databucs.models.outcomes import COMPETITIVE
from databucs.models.run_values import RESULT_SQL, RESULTS
from databucs.models.scoring import (
    CHAMPION_ALIAS,
    expected_value_sql,
    predictions_table,
    register_run_values,
    score_outcomes,
)

DECILES = 10
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]  # to join the predictions back


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate expected run value.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train-seasons", type=parse_seasons, default=[2023, 2024, 2025])
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument(
        "--model-alias", default=CHAMPION_ALIAS, help="Registry alias of the models to test."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ml = f"{args.catalog}.{args.ml_schema}"
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)

    # 1. Run values from the training seasons
    register_run_values(spark, fact_pitch, args.train_seasons, "run_values_wide")

    # 2. Score the test season with each champion model
    pitches = spark.sql(
        f"""
        SELECT
          {", ".join([*PITCH_KEY, *features.LOOKUP_KEY_COLUMNS])},
          {features.request_columns_sql()}
        FROM {fact_pitch}
        WHERE ({COMPETITIVE}) AND season = {args.test_season}
        """
    )
    score_outcomes(fe, pitches, ml, PITCH_KEY, args.model_alias).createOrReplaceTempView(
        "predictions"
    )

    # 3. Combine into expected run value, next to what actually happened
    probabilities, expected = expected_value_sql()
    combined = spark.sql(
        f"""
        WITH actual AS (
          SELECT game_pk, at_bat_number, pitch_number, game_date, pitcher_id, batter_id,
            pitch_type, balls, strikes, outs, base_state, run_value,
            {RESULT_SQL} AS result
          FROM {fact_pitch}
          WHERE ({COMPETITIVE}) AND season = {args.test_season}
        ),
        probabilities AS (
          SELECT a.*, {probabilities}
          FROM actual a
          JOIN predictions USING (game_pk, at_bat_number, pitch_number)
        )
        SELECT p.*, v.* EXCEPT (balls, strikes, outs, base_state), {expected} AS expected_run_value
        FROM probabilities p
        JOIN run_values_wide v USING (balls, strikes, outs, base_state)
        """
    )
    table = predictions_table(ml, args.model_alias)
    combined.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    print(f"Wrote {spark.table(table).count()} rows to {table}")

    # 4. Evaluate
    with mlflow.start_run(run_name=f"outcome_expected_run_value_{args.model_alias}"):
        mlflow.log_params(
            {
                "model_alias": args.model_alias,
                "train_seasons": ",".join(map(str, args.train_seasons)),
                "test_season": args.test_season,
                "predictions_table": table,
            }
        )

        # Predicted vs actual share of each result
        share_items = ", ".join(
            f"avg(p_{r}) AS predicted_{r}, avg(int(result = '{r}')) AS actual_{r}" for r in RESULTS
        )
        shares = spark.sql(f"SELECT {share_items} FROM {table}").first()
        share_rows = [
            {
                "result": r,
                "predicted_share": shares[f"predicted_{r}"],
                "actual_share": shares[f"actual_{r}"],
            }
            for r in RESULTS
        ]
        import pandas as pd

        share_df = pd.DataFrame(share_rows)
        mlflow.log_table(share_df, "result_shares.json")
        print(share_df.to_string(index=False))

        # Expected run value vs a count and base-out baseline (average run value per state
        # in the training seasons), on pitches with a run_value (not walk-offs)
        errors = spark.sql(
            f"""
            WITH state_baseline AS (
              SELECT balls, strikes, outs, base_state, avg(run_value) AS baseline
              FROM {fact_pitch}
              WHERE ({COMPETITIVE}) AND season IN ({", ".join(map(str, args.train_seasons))})
                AND run_value IS NOT NULL
              GROUP BY ALL
            )
            SELECT
              count(*) AS pitches,
              avg(power(run_value - expected_run_value, 2)) AS mse_model,
              avg(power(run_value - b.baseline, 2)) AS mse_state_baseline,
              avg(power(run_value, 2)) AS mse_zero,
              corr(run_value, expected_run_value) AS corr_model,
              corr(run_value, b.baseline) AS corr_state_baseline,
              avg(expected_run_value) AS mean_expected,
              avg(run_value) AS mean_actual
            FROM {table}
            JOIN state_baseline b USING (balls, strikes, outs, base_state)
            WHERE run_value IS NOT NULL
            """
        ).first()
        for name, value in errors.asDict().items():
            mlflow.log_metric(name, float(value))
            print(f"{name}: {value}")

        # Calibration of expected run value by decile
        deciles = spark.sql(
            f"""
            SELECT decile, count(*) AS pitches, avg(expected_run_value) AS mean_expected,
              avg(run_value) AS mean_actual
            FROM (
              SELECT *, ntile({DECILES}) OVER (ORDER BY expected_run_value) AS decile
              FROM {table}
              WHERE run_value IS NOT NULL
            )
            GROUP BY decile
            ORDER BY decile
            """
        ).toPandas()
        mlflow.log_table(deciles, "expected_run_value_deciles.json")
        print(deciles.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.bar(
            [x - 0.2 for x in range(len(RESULTS))],
            share_df["predicted_share"],
            width=0.4,
            label="predicted",
        )
        ax.bar(
            [x + 0.2 for x in range(len(RESULTS))],
            share_df["actual_share"],
            width=0.4,
            label="actual",
        )
        ax.set_xticks(range(len(RESULTS)), RESULTS, rotation=45, ha="right")
        ax.set_ylabel(f"Share of pitches ({args.test_season})")
        ax.set_title("Pitch results: predicted vs actual")
        ax.legend()
        ax = axes[1]
        ax.plot(deciles["mean_expected"], deciles["mean_actual"], marker="o")
        low = float(min(deciles["mean_expected"].min(), deciles["mean_actual"].min()))
        high = float(max(deciles["mean_expected"].max(), deciles["mean_actual"].max()))
        ax.plot([low, high], [low, high], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Expected run value (decile mean)")
        ax.set_ylabel("Actual run value (decile mean)")
        ax.set_title("Expected run value calibration")
        fig.tight_layout()
        mlflow.log_figure(fig, "expected_run_value.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
