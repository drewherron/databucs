"""The outcome model as an MLflow pyfunc: the feature transform plus the LightGBM model.

Logged with FeatureEngineeringClient.log_model, so fe.score_batch looks up the batter,
arsenal and game state features itself; the caller passes the keys, the game date and the
pitch's own columns (features.PITCH_COLUMNS). predict() returns one probability column per
class, named p_<class>.
"""

from __future__ import annotations

import mlflow
import pandas as pd

from databucs.models import features


class OutcomePyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, name: str, classes: list[str], model_text: str):
        self.name = name
        self.classes = list(classes)
        self.model_text = model_text  # LightGBM's own text format, portable across versions
        self._booster = None

    def load_context(self, context) -> None:
        self._load()

    def _load(self):
        if self._booster is None:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=self.model_text)
        return self._booster

    def predict_proba(self, df: pd.DataFrame):
        import numpy as np

        raw = self._load().predict(features.transform(df))
        if raw.ndim == 1:  # binary: LightGBM returns P(class 1)
            raw = np.column_stack([1 - raw, raw])
        return raw

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        probs = self.predict_proba(model_input)
        return pd.DataFrame(
            probs, columns=[f"p_{c}" for c in self.classes], index=model_input.index
        )
