"""Backtest the recommender's core claim on the test season: within the same situation,
pitches the outcome models rate better give up fewer runs (databucs.models.backtest).

Reads the test predictions written by evaluate_expected_run_value for --model-alias and
logs the results to the outcome_models MLflow experiment.
"""

from __future__ import annotations

import argparse

from databucs.models import backtest
from databucs.models.scoring import CHAMPION_ALIAS, predictions_table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Within-situation backtest.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--model-alias", default=CHAMPION_ALIAS, help="Whose test predictions to backtest."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    predictions = predictions_table(f"{args.catalog}.{args.ml_schema}", args.model_alias)
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"

    import mlflow
    import pandas as pd
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    mlflow.set_experiment(args.experiment)

    folds = spark.sql(backtest.backtest_sql(predictions, fact_pitch)).toPandas()
    summary = pd.DataFrame(backtest.summarize(folds))
    print(summary.to_string(index=False))

    with mlflow.start_run(run_name=f"recommender_backtest_{args.model_alias}"):
        mlflow.log_params(
            {
                "model_alias": args.model_alias,
                "predictions_table": predictions,
                "min_pitches": backtest.MIN_PITCHES,
                "folds": backtest.FOLDS,
            }
        )
        for row in summary.to_dict("records"):
            for key, value in row.items():
                if key != "grouping":
                    mlflow.log_metric(f"{row['grouping']}_{key}", float(value))
        mlflow.log_table(summary, "backtest_summary.json")
        mlflow.log_table(folds, "backtest_folds.json")


if __name__ == "__main__":
    main()
