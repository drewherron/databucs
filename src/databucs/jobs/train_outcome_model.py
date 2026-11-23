"""Train, evaluate and register one pitch outcome model.

1. Training rows come from gold.fact_pitch (the pitch, the game state, the label) joined to
   the batter and arsenal feature tables with FeatureEngineeringClient.create_training_set
   (point-in-time lookups).
2. LightGBM is trained on the training seasons, with early stopping on a holdout of 10% of
   their games.
3. The test season is scored and compared with simple baselines: the overall class rates,
   the rates by count, and the rates by pitch type.
4. Everything goes to MLflow: parameters, metrics, calibration tables and plots, and
   feature importance.
5. The model is logged with fe.log_model, so it carries its feature lookups, and
   registered in Unity Catalog as <catalog>.<ml schema>.outcome_<name>. With --promote
   if_better it becomes @champion only if its test log loss beats the current champion's
   (same test season); otherwise it gets @challenger. --promote always skips the check, for
   when the model's inputs changed and the old champion can't stand in for it.

Run one model per invocation, e.g. --model swing. The train_outcome_models job runs all
five in parallel.
"""

from __future__ import annotations

import argparse
import gc
import os

from databucs.models import features
from databucs.models.outcomes import BY_NAME, COMPETITIVE, OutcomeModel
from databucs.models.scoring import CHAMPION_ALIAS

HOLDOUT_GAMES_MOD = 10  # games with game_pk % 10 == 0 form the early-stopping holdout
CHALLENGER_ALIAS = "challenger"
PROMOTE_CHOICES = ("if_better", "always")

LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}
# The swing model was still improving at 3,000 rounds; 0.1 instead of 0.05 stopped at 1,700
# but tested worse (log loss 0.4320 vs 0.4291), so it gets more rounds instead
MAX_ROUNDS = 8000
EARLY_STOPPING_ROUNDS = 100


# Columns that stay integers; everything numeric or boolean among the features becomes float32
INTEGER_COLUMNS = {"label", "game_pk"}


def compact_columns(schema) -> list:
    """Spark columns for the pandas frame: transform()'s inputs as compact types."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, NumericType

    wanted = {*features.TRANSFORM_INPUTS, *INTEGER_COLUMNS}
    columns = [F.year("game_date").alias("season")]
    for field in schema.fields:
        if field.name not in wanted:
            continue
        if field.name in INTEGER_COLUMNS:
            columns.append(F.col(field.name).cast("int"))
        elif isinstance(field.dataType, NumericType | BooleanType):
            columns.append(F.col(field.name).cast("float"))
        else:
            columns.append(F.col(field.name))
    return columns


def fit_sample_condition(fit_rows: int, max_fit_rows: int, test_season: int) -> str:
    """SQL condition keeping about max_fit_rows of fit_rows fitting rows, by whole games.

    Holdout and test rows are always kept. Applied in Spark, before toPandas, because
    loading the rows is the peak of the job's memory use.
    """
    if fit_rows <= max_fit_rows:
        return "TRUE"
    per_mille = int(1000 * max_fit_rows / fit_rows)
    return (
        f"(year(game_date) = {test_season} OR game_pk % {HOLDOUT_GAMES_MOD} = 0"
        f" OR pmod(hash(game_pk, 'fit'), 1000) < {per_mille})"
    )


def should_promote(
    promote: str,
    new_log_loss: float,
    new_test_season: int,
    champion_log_loss: float | None,
    champion_test_season: int | None,
) -> tuple[bool, str]:
    """Whether the new version takes @champion, and why (printed by the job)."""
    if promote == "always":
        return True, "--promote always"
    if champion_log_loss is None:
        return True, "no current champion"
    if champion_test_season != new_test_season:
        return False, (
            f"champion was tested on {champion_test_season}, not {new_test_season}; "
            "compare by hand or use --promote always"
        )
    verdict = "beats" if new_log_loss < champion_log_loss else "does not beat"
    return new_log_loss < champion_log_loss, (
        f"test log loss {new_log_loss:.5f} {verdict} the champion's {champion_log_loss:.5f}"
    )


def parse_seasons(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one pitch outcome model.")
    parser.add_argument("--model", required=True, choices=sorted(BY_NAME))
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--experiment", required=True, help="MLflow experiment path.")
    parser.add_argument("--train-seasons", type=parse_seasons, default=[2023, 2024, 2025])
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument(
        "--sample-percent",
        type=int,
        default=100,
        help="Use only this percent of games (by game_pk), for quick trial runs.",
    )
    parser.add_argument(
        "--max-fit-rows",
        type=int,
        default=1_500_000,
        help="Fit on at most about this many rows, sampled by game, to fit in memory.",
    )
    parser.add_argument("--promote", choices=PROMOTE_CHOICES, default="if_better")
    return parser.parse_args(argv)


def training_where(spec: OutcomeModel, seasons: list[int], sample_percent: int) -> str:
    season_list = ", ".join(str(s) for s in seasons)
    where = f"({COMPETITIVE}) AND ({spec.where}) AND season IN ({season_list})"
    if sample_percent < 100:
        where += f" AND pmod(hash(game_pk), 100) < {sample_percent}"
    return where


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = BY_NAME[args.model]
    ml = f"{args.catalog}.{args.ml_schema}"
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"
    registered_name = f"{ml}.outcome_{spec.name}"

    import lightgbm as lgb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import pandas as pd
    from databricks.feature_engineering import FeatureEngineeringClient
    from mlflow.tracking import MlflowClient
    from pyspark.sql import SparkSession
    from sklearn.metrics import roc_auc_score

    import databucs
    from databucs.models import evaluation
    from databucs.models.pyfunc import OutcomePyfunc

    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)

    # 1. Training rows with point-in-time features
    seasons = [*args.train_seasons, args.test_season]
    labels = spark.sql(
        features.labels_sql(
            fact_pitch, training_where(spec, seasons, args.sample_percent), spec.label
        )
    )
    training_set = fe.create_training_set(
        df=labels, feature_lookups=features.feature_lookups(ml), label="label"
    )
    # The same rows without the split-only columns, for fe.log_model: it records the
    # training set's columns as the model's inputs
    logged_training_set = fe.create_training_set(
        df=labels,
        feature_lookups=features.feature_lookups(ml),
        label="label",
        exclude_columns=features.TRAINING_ONLY_COLUMNS,
    )
    # Only the columns transform() needs, cast to compact types in Spark. Arrow then builds
    # float32 pandas columns instead of float64 (and instead of Python objects for nullable
    # booleans), which roughly halves memory. Free Edition compute can't be resized.
    raw = training_set.load_df()
    fit_rows = raw.filter(
        f"year(game_date) <> {args.test_season} AND game_pk % {HOLDOUT_GAMES_MOD} <> 0"
    ).count()
    sample = fit_sample_condition(fit_rows, args.max_fit_rows, args.test_season)
    data = raw.filter(sample).select(*compact_columns(raw.schema)).toPandas()
    print(f"{spec.name}: {len(data)} rows for seasons {seasons}")

    y = data["label"].to_numpy()
    season = data["season"].to_numpy()
    game_pk = data["game_pk"].to_numpy()
    is_test = season == args.test_season
    is_holdout = ~is_test & (game_pk % HOLDOUT_GAMES_MOD == 0)
    is_fit = ~is_test & ~is_holdout
    n_classes = len(spec.classes)

    X = features.transform(data)
    del data
    gc.collect()

    # Small frames for the baselines, the test rows for scoring, then LightGBM's binned
    # datasets; after that the full feature frame can go
    baseline_columns = ["balls", "strikes", "pitch_type"]
    baseline_fit = X.loc[is_fit, baseline_columns].copy()
    X_test = X[is_test]
    fit_set = lgb.Dataset(X[is_fit], y[is_fit], categorical_feature=features.CATEGORICAL_FEATURES)
    holdout_set = lgb.Dataset(
        X[is_holdout],
        y[is_holdout],
        categorical_feature=features.CATEGORICAL_FEATURES,
        reference=fit_set,
    )
    fit_set.construct()
    holdout_set.construct()
    del X
    gc.collect()

    # 2. Train
    params = dict(LGB_PARAMS)
    if spec.is_binary:
        params.update(objective="binary", metric="binary_logloss")
    else:
        params.update(objective="multiclass", num_class=n_classes, metric="multi_logloss")

    with mlflow.start_run(run_name=f"outcome_{spec.name}") as run:
        mlflow.log_params(
            {
                **{f"lgb_{k}": v for k, v in params.items()},
                "model": spec.name,
                "classes": ",".join(spec.classes),
                "train_seasons": ",".join(map(str, args.train_seasons)),
                "test_season": args.test_season,
                "sample_percent": args.sample_percent,
                "max_fit_rows": args.max_fit_rows,
                "fit_rows": int(is_fit.sum()),
                "holdout_rows": int(is_holdout.sum()),
                "test_rows": int(is_test.sum()),
                "features": len(features.FEATURES),
            }
        )
        booster = lgb.train(
            params,
            fit_set,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[holdout_set],
            valid_names=["holdout"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)],
        )
        mlflow.log_metric("best_iteration", booster.best_iteration)

        # 3. Test-season metrics against baselines
        y_test = y[is_test]
        raw_test = booster.predict(X_test, num_iteration=booster.best_iteration)
        model_test = np.column_stack([1 - raw_test, raw_test]) if raw_test.ndim == 1 else raw_test

        def baseline(columns):
            return evaluation.frequency_baseline(
                baseline_fit[columns], y[is_fit], X_test[columns], n_classes
            )

        candidates = {
            "model": model_test,
            "baseline_prior": baseline([]),
            "baseline_count": baseline(["balls", "strikes"]),
            "baseline_pitch_type": baseline(["pitch_type"]),
        }
        results = []
        for name, probs in candidates.items():
            row = {
                "candidate": name,
                "log_loss": evaluation.log_loss(y_test, probs),
                "brier": evaluation.brier(y_test, probs),
            }
            if spec.is_binary:
                row["auc"] = float(roc_auc_score(y_test, probs[:, 1]))
            results.append(row)
            for metric, value in row.items():
                if metric != "candidate":
                    mlflow.log_metric(f"test_{metric}_{name}", value)
        results_df = pd.DataFrame(results)
        mlflow.log_table(results_df, "test_metrics.json")
        print(results_df.to_string(index=False))

        # Calibration on the test season, one curve per class (the positive class if binary)
        model_probs = candidates["model"]
        classes_to_check = [1] if spec.is_binary else list(range(n_classes))
        fig, ax = plt.subplots(figsize=(6, 6))
        for k in classes_to_check:
            table = evaluation.calibration_table((y_test == k).astype(float), model_probs[:, k])
            ece = evaluation.expected_calibration_error(table)
            mlflow.log_metric(f"test_ece_{spec.classes[k]}", ece)
            mlflow.log_table(table, f"calibration_{spec.classes[k]}.json")
            ax.plot(
                table["mean_predicted"],
                table["observed_rate"],
                marker="o",
                label=f"{spec.classes[k]} (ECE {ece:.4f})",
            )
        top = float(max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.plot([0, top], [0, top], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel(f"Observed rate ({args.test_season})")
        ax.set_title(f"outcome_{spec.name}: calibration")
        ax.legend()
        mlflow.log_figure(fig, "calibration.png")
        plt.close(fig)

        # Feature importance (total gain)
        importance = (
            pd.DataFrame(
                {
                    "feature": booster.feature_name(),
                    "gain": booster.feature_importance("gain"),
                    "splits": booster.feature_importance("split"),
                }
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        importance["gain_share"] = importance["gain"] / importance["gain"].sum()
        mlflow.log_table(importance, "feature_importance.json")
        top_features = importance.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.barh(top_features["feature"], top_features["gain_share"])
        ax.set_xlabel("Share of total gain")
        ax.set_title(f"outcome_{spec.name}: top features")
        fig.tight_layout()
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        # 5. Log with feature lineage and register
        model = OutcomePyfunc(spec.name, list(spec.classes), booster.model_to_string())
        fe.log_model(
            model=model,
            artifact_path="model",
            flavor=mlflow.pyfunc,
            training_set=logged_training_set,
            registered_model_name=registered_name,
            code_paths=[os.path.dirname(databucs.__file__)],
            extra_pip_requirements=[f"lightgbm=={lgb.__version__}"],
        )

    client = MlflowClient(registry_uri="databricks-uc")
    # Unity Catalog's registry only filters versions by name, so match the run here
    version = max(
        (
            v
            for v in client.search_model_versions(f"name = '{registered_name}'")
            if v.run_id == run.info.run_id
        ),
        key=lambda v: int(v.version),
    ).version
    client.update_registered_model(registered_name, description=spec.description)

    # 6. Champion or challenger
    champion_log_loss = champion_test_season = None
    try:
        champion = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
    except Exception:  # no alias yet (MlflowException)
        champion = None
    if champion is not None and champion.version != version:
        champion_run = client.get_run(champion.run_id).data
        champion_log_loss = champion_run.metrics.get("test_log_loss_model")
        champion_test_season = int(champion_run.params.get("test_season", 0)) or None
    new_log_loss = float(results_df.loc[results_df["candidate"] == "model", "log_loss"].iloc[0])
    promote, reason = should_promote(
        args.promote, new_log_loss, args.test_season, champion_log_loss, champion_test_season
    )
    alias = CHAMPION_ALIAS if promote else CHALLENGER_ALIAS
    client.set_registered_model_alias(registered_name, alias, version)
    print(f"Registered {registered_name} version {version} as @{alias}: {reason}")


if __name__ == "__main__":
    main()
