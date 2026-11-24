"""Build, check and serve the real-time pitch recommender.

1. Snapshot, as of --as-of-date (default today), of everything the recommender looks up,
   built with the batch job's own queries (databucs.recommender.inputs): batters (names,
   strike zones, sides), pitchers, arsenals, the latest batter profiles, every pitcher's
   usual mix for every count, and the run values. Saved as parquet artifacts.
2. The five @champion outcome models' LightGBM boosters, pulled out of their logged
   pyfuncs.
3. Consistency check: when the batch table has a series starting on the as-of date, a
   sample of its situations is recomputed with the pandas engine
   (databucs.recommender.serving) and must match.
4. Log the RecommenderPyfunc with both as artifacts, register it as
   <ml schema>.pitch_recommender and point @champion at it.
5. Create the serving endpoint, or move it to the new version, and send it one request.

Run it whenever the features should be refreshed (daily in season, or before a series).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from dataclasses import asdict

from databucs.ingestion.dates import optional_date
from databucs.jobs.train_outcome_model import parse_seasons
from databucs.models import features
from databucs.models.outcomes import MODELS
from databucs.models.scoring import CHAMPION_ALIAS
from databucs.recommender import inputs, options

MODEL_NAME = "pitch_recommender"
CHECK_SITUATIONS = 20
TOLERANCE = 1e-6  # runs per pitch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and serve the recommender.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--as-of-date", type=optional_date, default=None)
    parser.add_argument("--run-value-seasons", type=parse_seasons, default=[2023, 2024, 2025])
    return parser.parse_args(argv)


def snapshot_tables(spark, args, as_of: dt.date) -> dict:
    """The snapshot as pandas frames (see serving.SNAPSHOT_TABLES)."""
    from databucs.models.scoring import register_run_values

    silver = f"`{args.catalog}`.`{args.silver_schema}`"
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"
    ml = f"{args.catalog}.{args.ml_schema}"
    arsenal_table, profile_table = f"{ml}.pitcher_arsenal_features", f"{ml}.batter_profile"
    players = f"{silver}.players"
    target = inputs.Target(inputs.PIRATES_TEAM_ID, 0, as_of, "snapshot")
    history = (
        f"game_date >= date_sub(DATE'{as_of}', {inputs.HISTORY_DAYS}) AND game_date < DATE'{as_of}'"
    )

    # Everyone active in the trailing window, and every count
    spark.sql(
        "SELECT DISTINCT pitcher_id, false AS is_probable_starter "
        f"FROM {fact_pitch} WHERE {history}"
    ).createOrReplaceTempView("target_pitchers")
    spark.sql(
        "SELECT DISTINCT batter_id, CAST(NULL AS INT) AS lineup_spot "
        f"FROM {fact_pitch} WHERE {history}"
    ).createOrReplaceTempView("target_lineup")
    spark.createDataFrame(
        [(b, s) for b in range(4) for s in range(3)], "balls int, strikes int"
    ).createOrReplaceTempView("scenarios")
    register_run_values(spark, fact_pitch, args.run_value_seasons, "run_values_wide")

    profile_columns = ", ".join([*features.BATTER_FEATURES, *features.BATTER_CELL_COUNTS])
    queries = {
        "batters": f"SELECT batter_id, batter_name, bats, sz_top, sz_bot FROM "
        f"({inputs.batters_sql(fact_pitch, players, target)})",
        "sides": inputs.sides_sql(fact_pitch, target),
        "pitchers": f"SELECT pitcher_id, pitcher_name, pitcher_throws FROM "
        f"({inputs.pitchers_sql(fact_pitch, players, target)})",
        "arsenal": inputs.arsenal_sql(arsenal_table, fact_pitch, target),
        "batter_features": f"""
            SELECT batter_id, {profile_columns}
            FROM {profile_table}
            WHERE as_of_date <= DATE'{as_of}'
            QUALIFY row_number() OVER (PARTITION BY batter_id ORDER BY as_of_date DESC) = 1""",
        "usage_land": inputs.usage_sql(fact_pitch, target, "land"),
        "usage_zone": inputs.usage_sql(fact_pitch, target, "zone"),
        "run_values": "SELECT * FROM run_values_wide",
    }
    tables = {}
    for name, sql in queries.items():
        tables[name] = spark.sql(sql).toPandas()
        # Spark Connect attaches query metrics here, which parquet can't serialize
        tables[name].attrs = {}
    # Arsenal rows of pitchers who haven't pitched in the window are stale
    active = tables["arsenal"]["pitcher_id"].isin(tables["pitchers"]["pitcher_id"])
    tables["arsenal"] = tables["arsenal"][active].reset_index(drop=True)
    for name, frame in tables.items():
        print(f"snapshot {name}: {len(frame)} rows", flush=True)
    return tables


def champion_boosters(ml: str) -> dict:
    """{model name: {"classes", "model_text"}} from each @champion outcome model. They were
    logged with fe.log_model, which keeps the original pyfunc as a nested raw model."""
    import mlflow

    boosters = {}
    for spec in MODELS:
        folder = mlflow.artifacts.download_artifacts(
            f"models:/{ml}.outcome_{spec.name}@{CHAMPION_ALIAS}"
        )
        raw = None
        for root, _dirs, files in os.walk(folder):
            if "python_model.pkl" in files or "python_model.cloudpickle" in files:
                raw = root
                break
        if raw is None:
            raise RuntimeError(f"no python model inside {spec.name}'s artifacts")
        model = mlflow.pyfunc.load_model(raw).unwrap_python_model()
        boosters[spec.name] = {"classes": list(model.classes), "model_text": model.model_text}
    return boosters


def check_against_batch(spark, ml: str, as_of: dt.date, snapshot, models) -> None:
    """Recompute a sample of the batch table's situations for this date and compare."""
    import numpy as np

    from databucs.recommender.serving import recommend

    batch = spark.sql(
        f"""
        SELECT pitcher_id, batter_id, scenario, pitch_type, zone, expected_run_value,
          vs_usual, tier
        FROM {ml}.pitch_option_scores
        WHERE series_start = DATE'{as_of}'
        """
    ).toPandas()
    if batch.empty:
        print(f"No batch rows for series starting {as_of}: consistency check skipped")
        return
    scenarios = {s.name: s for s in options.SCENARIOS}
    situations = (
        batch[["pitcher_id", "batter_id", "scenario"]]
        .drop_duplicates()
        .sample(n=CHECK_SITUATIONS, random_state=7, replace=False)
    )
    worst = 0.0
    for row in situations.itertuples():
        s = scenarios[row.scenario]
        request = {
            "pitcher_id": row.pitcher_id,
            "batter_id": row.batter_id,
            **{k: v for k, v in asdict(s).items() if k not in ("name", "description")},
        }
        served = recommend(snapshot, models, request).set_index(["pitch_type", "zone"])
        expected = batch[
            (batch.pitcher_id == row.pitcher_id)
            & (batch.batter_id == row.batter_id)
            & (batch.scenario == row.scenario)
        ].set_index(["pitch_type", "zone"])
        if set(served.index) != set(expected.index):
            raise RuntimeError(f"options differ for {request}")
        joined = expected.join(served, rsuffix="_served")
        diff = np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        joined["expected_run_value"] - joined["expected_run_value_served"],
                        joined["vs_usual"] - joined["vs_usual_served"],
                    ]
                )
            )
        )
        worst = max(worst, float(diff))
        if diff > TOLERANCE or (joined["tier"] != joined["tier_served"]).any():
            raise RuntimeError(f"served and batch differ by {diff} for {request}")
    print(
        f"Consistency check: {CHECK_SITUATIONS} situations match (largest difference {worst:.2e})"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    as_of = args.as_of_date or dt.date.today()
    ml = f"{args.catalog}.{args.ml_schema}"
    registered_name = f"{ml}.{MODEL_NAME}"

    import lightgbm
    import mlflow
    import numpy
    import pandas as pd
    import pyarrow
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    from mlflow.types import ColSpec, Schema
    from pyspark.sql import SparkSession

    import databucs
    from databucs.models.pyfunc import OutcomePyfunc
    from databucs.recommender.pyfunc import OPTIONAL_COLUMNS, REQUEST_COLUMNS, RecommenderPyfunc
    from databucs.recommender.serving import Snapshot

    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(args.experiment)

    # 1-2. Snapshot and boosters
    tables = snapshot_tables(spark, args, as_of)
    boosters = champion_boosters(ml)
    snapshot = Snapshot(str(as_of), tables)
    models = {n: OutcomePyfunc(n, b["classes"], b["model_text"]) for n, b in boosters.items()}

    # 3. Same numbers as the batch job?
    check_against_batch(spark, ml, as_of, snapshot, models)

    # 4. Log and register
    folder = tempfile.mkdtemp()
    snapshot_dir = os.path.join(folder, "snapshot")
    os.makedirs(snapshot_dir)
    for name, frame in tables.items():
        frame.to_parquet(os.path.join(snapshot_dir, f"{name}.parquet"), index=False)
    with open(os.path.join(snapshot_dir, "meta.json"), "w") as f:
        json.dump({"as_of_date": str(as_of)}, f)
    boosters_path = os.path.join(folder, "outcome_models.json")
    with open(boosters_path, "w") as f:
        json.dump(boosters, f)

    # An example request: the pitcher and batter with the most data in the snapshot, 1-2 with
    # a runner on third
    pitcher_id = int(tables["arsenal"].groupby("pitcher_id")["pitches_thrown"].sum().idxmax())
    batters = tables["batters"].merge(tables["batter_features"], on="batter_id")
    batter_id = int(batters.sort_values("pitches_seen").iloc[-1]["batter_id"])
    example = pd.DataFrame(
        [
            {
                "pitcher_id": pitcher_id,
                "batter_id": batter_id,
                "balls": 1,
                "strikes": 2,
                "outs": 1,
                "base_state": "__3",
                "inning": 7,
                "bat_score_diff": -1,
                "times_through_order": 3,
            }
        ]
    )
    wrapper = RecommenderPyfunc()
    wrapper.snapshot, wrapper.models = snapshot, models
    params = {"top_n": 10, "miss_sd_ft": options.MISS_SD_FT}
    output = wrapper.predict(None, example, params)
    print(output.head(10).to_string(index=False))
    signature = infer_signature(example, output, params=params)
    signature.inputs = Schema(
        [ColSpec(t, n, required=n not in OPTIONAL_COLUMNS) for n, t in REQUEST_COLUMNS.items()]
    )

    with mlflow.start_run(run_name="pitch_recommender") as run:
        mlflow.log_params(
            {
                "as_of_date": str(as_of),
                "miss_sd_ft": options.MISS_SD_FT,
                **{f"snapshot_{n}_rows": len(t) for n, t in tables.items()},
            }
        )
        mlflow.pyfunc.log_model(
            name="recommender",
            python_model=RecommenderPyfunc(),
            artifacts={"snapshot": snapshot_dir, "outcome_models": boosters_path},
            code_paths=[os.path.dirname(databucs.__file__)],
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"lightgbm=={lightgbm.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={numpy.__version__}",
                f"pyarrow=={pyarrow.__version__}",
            ],
            signature=signature,
            input_example=example,
            registered_model_name=registered_name,
        )
    client = MlflowClient(registry_uri="databricks-uc")
    version = max(
        (
            v
            for v in client.search_model_versions(f"name = '{registered_name}'")
            if v.run_id == run.info.run_id
        ),
        key=lambda v: int(v.version),
    ).version
    client.set_registered_model_alias(registered_name, CHAMPION_ALIAS, version)
    client.update_registered_model(
        registered_name,
        description="Real-time pitch recommender: options for a pitcher, batter and situation.",
    )
    print(f"Registered {registered_name} version {version} (features as of {as_of})")

    # 5. Serve it
    w = WorkspaceClient()
    entity = ServedEntityInput(
        entity_name=registered_name,
        entity_version=str(version),
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
    existing = {e.name for e in w.serving_endpoints.list()}
    timeout = dt.timedelta(minutes=50)
    if args.endpoint_name in existing:
        print(f"Moving endpoint {args.endpoint_name} to version {version}")
        w.serving_endpoints.update_config_and_wait(
            args.endpoint_name, served_entities=[entity], timeout=timeout
        )
    else:
        print(f"Creating endpoint {args.endpoint_name}")
        w.serving_endpoints.create_and_wait(
            args.endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[entity]),
            timeout=timeout,
        )
    answer = w.serving_endpoints.query(
        args.endpoint_name,
        dataframe_records=example.to_dict("records"),
    )
    print(f"Endpoint answered with {len(answer.predictions)} options")


if __name__ == "__main__":
    main()
