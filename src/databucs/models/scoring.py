"""Score rows with the five @champion outcome models and turn them into expected run value.

Shared by the evaluation job (real 2026 pitches) and the recommender (hypothetical
pitches), so both combine the models the same way:

1. score_outcomes: each model scores the rows with fe.score_batch (which adds the batter
   and arsenal lookups), and the five predictions are joined back into one struct `p`.
2. register_run_values: run values per state and result, one column per result.
3. expected_value_sql: SELECT items for each result's probability and the expected run
   value, over rows that have `p` and the run value columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from databucs.models.outcomes import MODELS
from databucs.models.run_values import (
    RESULT_PROBABILITY_SQL,
    RESULTS,
    prediction_struct_type,
    run_values_sql,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

CHAMPION_ALIAS = "champion"
STATE_KEY = ["balls", "strikes", "outs", "base_state"]


def score_outcomes(
    fe, df: DataFrame, ml: str, keys: list[str], alias: str = CHAMPION_ALIAS
) -> DataFrame:
    """`keys` plus a struct column `p` with each model's prediction struct (the versions
    with registry alias `alias`, @champion by default).

    `df` needs the models' inputs (features.LOOKUP_KEY_COLUMNS, PITCH_COLUMNS and
    STATE_COLUMNS) and `keys`, which must identify each row.
    """
    from pyspark.sql import functions as F

    scored = df.select(*keys)
    for model in MODELS:
        prediction = fe.score_batch(
            model_uri=f"models:/{ml}.outcome_{model.name}@{alias}",
            df=df,
            result_type=prediction_struct_type(model.name),
        )
        scored = scored.join(prediction.select(*keys, F.col("prediction").alias(model.name)), keys)
    return scored.select(*keys, F.struct(*[m.name for m in MODELS]).alias("p"))


def register_run_values(spark: SparkSession, fact_pitch: str, seasons: list[int], view: str):
    """Temp view `view`: one row per state (STATE_KEY) with a column rv_<result> each."""
    spark.sql(run_values_sql(fact_pitch, seasons)).createOrReplaceTempView(f"{view}_long")
    wide = ",\n  ".join(
        f"max(CASE WHEN result = '{r}' THEN run_value END) AS rv_{r}" for r in RESULTS
    )
    spark.sql(
        f"SELECT {', '.join(STATE_KEY)}, {wide} FROM {view}_long GROUP BY ALL"
    ).createOrReplaceTempView(view)


def expected_value_sql() -> tuple[str, str]:
    """SQL select items for each result's probability (p_<result>), and the expected run
    value. The probabilities read `p`; the expected value reads p_<result> and rv_<result>,
    so it goes one query level above the probabilities."""
    probabilities = ",\n  ".join(f"{RESULT_PROBABILITY_SQL[r]} AS p_{r}" for r in RESULTS)
    expected = " + ".join(f"p_{r} * rv_{r}" for r in RESULTS)
    return probabilities, expected
