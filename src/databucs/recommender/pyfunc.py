"""The real-time recommender as an MLflow pyfunc, for a Model Serving endpoint.

Artifacts (written by jobs/register_recommender.py):
- snapshot/<table>.parquet for each of serving.SNAPSHOT_TABLES, plus snapshot/meta.json
  with the as-of date
- outcome_models.json: each outcome model's classes and LightGBM model text

Input: one row per question (REQUEST_COLUMNS). Output: one row per option, with
request_index pointing back at the input row, ranked within each request. Params:
top_n (options per request, 0 = all) and miss_sd_ft.
"""

from __future__ import annotations

import json
import os

import mlflow
import pandas as pd

from databucs.recommender import options

REQUEST_COLUMNS = {
    "pitcher_id": "long",
    "batter_id": "long",
    "balls": "long",
    "strikes": "long",
    "outs": "long",
    "base_state": "string",
    "inning": "long",  # optional, default 5
    "bat_score_diff": "long",  # optional: batting team's score minus the fielding team's
    "times_through_order": "long",  # optional, default 2
}
OPTIONAL_COLUMNS = {"inning", "bat_score_diff", "times_through_order"}

OUTPUT_COLUMNS = [
    "request_index",
    "option_rank",
    "tier",
    "pitch_name",
    "pitch_type",
    "location",
    "zone",
    "expected_run_value",
    "expected_run_value_low",
    "expected_run_value_high",
    "vs_usual",
    "usual_share",
    "usual_expected_run_value",
    "p_ball",
    "p_called_strike",
    "p_whiff",
    "p_foul",
    "p_out",
    "p_single",
    "p_double",
    "p_triple",
    "p_home_run",
    "p_hit_by_pitch",
    "pitcher_name",
    "batter_name",
    "batter_stands",
    "as_of_date",
    "error",
]
DEFAULT_TOP_N = 10


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        from databucs.models.pyfunc import OutcomePyfunc
        from databucs.recommender.serving import SNAPSHOT_TABLES, Snapshot

        folder = context.artifacts["snapshot"]
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        tables = {t: pd.read_parquet(os.path.join(folder, f"{t}.parquet")) for t in SNAPSHOT_TABLES}
        self.snapshot = Snapshot(meta["as_of_date"], tables)
        with open(context.artifacts["outcome_models"]) as f:
            saved = json.load(f)
        self.models = {
            name: OutcomePyfunc(name, m["classes"], m["model_text"]) for name, m in saved.items()
        }

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from databucs.recommender.serving import RequestError, recommend

        params = params or {}
        top_n = int(params.get("top_n", DEFAULT_TOP_N))
        miss_sd_ft = float(params.get("miss_sd_ft", options.MISS_SD_FT))
        frames = []
        for index, row in model_input.reset_index(drop=True).iterrows():
            request = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            try:
                result = recommend(self.snapshot, self.models, request, miss_sd_ft)
            except RequestError as e:  # answer the other requests; report this one
                result = pd.DataFrame({"error": [str(e)], "as_of_date": [self.snapshot.as_of_date]})
            if top_n > 0 and "option_rank" in result:
                result = result[result["option_rank"] <= top_n]
            result.insert(0, "request_index", index)
            frames.append(result)
        out = pd.concat(frames, ignore_index=True)
        for column in OUTPUT_COLUMNS:
            if column not in out:
                out[column] = None
        return out[OUTPUT_COLUMNS]
