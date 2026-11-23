"""Inputs to the outcome models.

A training or scoring row starts with the pitch's own columns (PITCH_COLUMNS, from
fact_pitch in training, or a hypothetical pitch when the recommender scores options).
Feature Engineering then adds the batter profile, the pitcher's arsenal and the game state
through point-in-time lookups (feature_lookups). Finally transform() derives a few
features and returns exactly the model's input columns, in a fixed order with fixed
category levels. Training and the logged pyfunc model both call transform(), so the
features are computed the same way in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from databucs.features import batter_profile, game_state, pitcher_arsenal

if TYPE_CHECKING:
    import pandas as pd

KEY_COLUMNS = ["game_pk", "at_bat_number", "pitch_number"]

# The pitch's own columns, as SQL over fact_pitch. Horizontal values are flipped for
# right-handed pitchers so positive is arm side, like the arsenal table.
PITCH_COLUMNS = {
    "pitch_type": "pitch_type",
    "pitch_group": "pitch_group",
    "pitcher_throws": "pitcher_throws",
    "batter_stands": "batter_stands",
    "is_in_zone": "is_in_zone",
    "plate_x": "plate_x",
    "plate_z": "plate_z",
    "sz_top": "sz_top",
    "sz_bot": "sz_bot",
    "pitch_velocity": "release_speed",
    "pitch_spin_rate": "release_spin_rate",
    "pitch_horizontal_break": "pfx_x * 12 * if(pitcher_throws = 'R', -1, 1)",
    "pitch_induced_vertical_break": "pfx_z * 12",
    "pitch_release_side": "release_pos_x * if(pitcher_throws = 'R', -1, 1)",
    "pitch_release_height": "release_pos_z",
    "pitch_extension": "release_extension",
    "pitch_arm_angle": "arm_angle",
}

# Columns every row needs besides the pitch's own: keys for the lookups
LOOKUP_KEY_COLUMNS = [*KEY_COLUMNS, "game_date", "batter_id", "pitcher_id"]

CATEGORIES = {
    "pitch_type": [
        "FF",
        "SI",
        "FC",
        "SL",
        "ST",
        "SV",
        "CU",
        "KC",
        "CS",
        "CH",
        "FS",
        "FO",
        "SC",
        "KN",
        "EP",
        "FA",
        "UN",
    ],  # fmt: skip
    "pitch_group": ["fastball", "breaking", "offspeed", "other"],
    "pitcher_throws": ["L", "R"],
    "batter_stands": ["L", "R"],
    "base_state": ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"],
    "count_group": list(batter_profile.COUNT_GROUPS),
}

GAME_STATE_FEATURES = [
    "balls",
    "strikes",
    "outs",
    "inning",
    "bat_score_diff",
    "times_through_order",
    "runners_on",
    "runner_in_scoring_position",
    "is_late_and_close",
    "is_first_pitch_of_pa",
    "is_platoon_same_side",
]
CELL_STATS = ("swing_rate", "whiff_rate", "xwoba_contact")
BATTER_FEATURES = [
    "pitches_seen",
    "plate_appearances",
    "zone_swing_rate",
    "chase_rate",
    "whiff_rate",
    "xwoba_contact",
    "strikeout_rate",
    "walk_rate",
    "bat_speed",
    "swing_length",
    *[f"{cell}_{stat}" for cell, _ in batter_profile.cells() for stat in CELL_STATS],
]
# Looked up only to build cell_pitches below, not model inputs themselves
BATTER_CELL_COUNTS = [f"{cell}_pitches" for cell, _ in batter_profile.cells()]
ARSENAL_FEATURES = [
    "pitches_thrown",
    "usage_rate",
    *[name for name, _ in pitcher_arsenal.MEASURES],
    *[f"{name}_diff_from_fastball" for name in pitcher_arsenal.FASTBALL_DIFFS],
]
PITCH_FEATURES = [
    "is_in_zone",
    "plate_x_toward_batter",
    "plate_z",
    "plate_height_in_zone",
    "pitch_velocity",
    "pitch_spin_rate",
    "pitch_horizontal_break",
    "pitch_induced_vertical_break",
    "pitch_release_side",
    "pitch_release_height",
    "pitch_extension",
    "pitch_arm_angle",
    "pitch_velocity_diff_from_fastball",
    "pitch_horizontal_break_diff_from_fastball",
    "pitch_induced_vertical_break_diff_from_fastball",
]
# The batter's rates in this pitch's cell (pitch group x region x count group)
CELL_FEATURES = ["cell_pitches", "cell_swing_rate", "cell_whiff_rate", "cell_xwoba_contact"]

# Every column transform() reads: the pitch's own columns plus the looked-up features
TRANSFORM_INPUTS = [
    *PITCH_COLUMNS,
    *GAME_STATE_FEATURES,
    "base_state",
    "count_group",
    *BATTER_FEATURES,
    *BATTER_CELL_COUNTS,
    *ARSENAL_FEATURES,
]

NUMERIC_FEATURES = [
    *GAME_STATE_FEATURES,
    *BATTER_FEATURES,
    *ARSENAL_FEATURES,
    *PITCH_FEATURES,
    *CELL_FEATURES,
]
CATEGORICAL_FEATURES = list(CATEGORIES)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def labels_sql(fact_pitch: str, where: str, label: str) -> str:
    """SQL for training rows: keys, the pitch's own columns and the label.

    Every non-label column here becomes part of the logged model's input (fe.log_model
    records them), so nothing training-only belongs here: the training job derives the
    season from game_date itself.
    """
    pitch = ",\n  ".join(f"{expr} AS {name}" for name, expr in PITCH_COLUMNS.items())
    return f"""
SELECT
  {", ".join(LOOKUP_KEY_COLUMNS)},
  {pitch},
  {label} AS label
FROM {fact_pitch}
WHERE {where}
"""


def feature_lookups(ml_schema: str) -> list:
    """FeatureLookups for the three feature tables in `ml_schema` (catalog.schema)."""
    from databricks.feature_engineering import FeatureLookup

    return [
        FeatureLookup(
            table_name=f"{ml_schema}.{batter_profile.TABLE}",
            lookup_key="batter_id",
            timestamp_lookup_key="game_date",
            feature_names=BATTER_FEATURES + BATTER_CELL_COUNTS,
        ),
        FeatureLookup(
            table_name=f"{ml_schema}.{pitcher_arsenal.TABLE}",
            lookup_key=["pitcher_id", "pitch_type"],
            timestamp_lookup_key="game_date",
            feature_names=ARSENAL_FEATURES,
        ),
        FeatureLookup(
            table_name=f"{ml_schema}.{game_state.TABLE}",
            lookup_key=KEY_COLUMNS,
            feature_names=[*GAME_STATE_FEATURES, "base_state", "count_group"],
        ),
    ]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features and return the model's input columns (FEATURES), in order.

    `df` has the pitch's own columns plus the looked-up features. Missing lookups (a
    debut, a new pitch type) stay NaN, which LightGBM handles.
    """
    import numpy as np
    import pandas as pd

    out: dict[str, object] = {}  # built as a dict, then one DataFrame (much faster)
    numeric = lambda col: pd.to_numeric(df[col], errors="coerce").astype("float32")  # noqa: E731

    # Location: positive x is toward the batter; height as a fraction of the batter's zone
    toward_batter = np.where(df["batter_stands"] == "R", -1.0, 1.0)
    out["plate_x_toward_batter"] = numeric("plate_x") * toward_batter
    out["plate_z"] = numeric("plate_z")
    zone_height = numeric("sz_top") - numeric("sz_bot")
    out["plate_height_in_zone"] = (out["plate_z"] - numeric("sz_bot")) / zone_height

    for col in [*GAME_STATE_FEATURES, *BATTER_FEATURES, *ARSENAL_FEATURES]:
        out[col] = numeric(col)
    out["is_in_zone"] = numeric("is_in_zone")
    for col in PITCH_FEATURES:
        if col.startswith("pitch_") and not col.endswith("_from_fastball"):
            out[col] = numeric(col)

    # This pitch compared with the pitcher's primary fastball. The arsenal row stores this
    # pitch type's average difference, so the fastball's average is (average - difference).
    for name in pitcher_arsenal.FASTBALL_DIFFS:
        fastball = numeric(name) - numeric(f"{name}_diff_from_fastball")
        out[f"pitch_{name}_diff_from_fastball"] = numeric(f"pitch_{name}") - fastball

    # The batter's rates in the cell this pitch falls in
    region = np.where(df["is_in_zone"].astype("boolean").fillna(False), "zone", "chase")
    cell = pd.Series(
        df["pitch_group"].astype("string")
        + "_"
        + region
        + "_"
        + df["count_group"].astype("string"),
        index=df.index,
    )
    for feature, stat in [
        ("cell_pitches", "pitches"),
        ("cell_swing_rate", "swing_rate"),
        ("cell_whiff_rate", "whiff_rate"),
        ("cell_xwoba_contact", "xwoba_contact"),
    ]:
        values = np.full(len(df), np.nan, dtype="float32")
        for name, _ in batter_profile.cells():
            column = f"{name}_{stat}"
            if column in df:
                mask = (cell == name).fillna(False).to_numpy()
                values[mask] = pd.to_numeric(df[column], errors="coerce").to_numpy()[mask]
        out[feature] = values

    for col, levels in CATEGORIES.items():
        # Values outside the known levels (e.g. a new pitch type code) become missing
        known = df[col].where(df[col].isin(levels))
        out[col] = pd.Categorical(known, categories=levels)

    return pd.DataFrame(out, index=df.index)[FEATURES]
