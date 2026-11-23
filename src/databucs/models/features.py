"""Inputs to the outcome models.

A training or scoring row starts with the pitch's own columns (PITCH_COLUMNS) and the game
state before it (STATE_COLUMNS). In training both come from fact_pitch; when the
recommender scores options they describe a hypothetical pitch and situation, so they are
request-time inputs rather than lookups. Feature Engineering then adds the batter profile
and the pitcher's arsenal through point-in-time lookups (feature_lookups). Finally
transform() derives the remaining features and returns exactly the model's input columns,
in a fixed order with fixed category levels. Training and the logged pyfunc model both call
transform(), so the features are computed the same way in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from databucs.features import batter_profile, pitcher_arsenal

if TYPE_CHECKING:
    import pandas as pd

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

# The situation before the pitch, as SQL over fact_pitch. bat_score_diff is the batting
# team's score minus the fielding team's; base_state is like '1_3' (runners on first and
# third).
STATE_COLUMNS = {
    "balls": "balls",
    "strikes": "strikes",
    "outs": "outs",
    "inning": "inning",
    "bat_score_diff": "bat_score_diff",
    "times_through_order": "times_through_order",
    "base_state": "base_state",
}

# Keys for the lookups (pitch_type, the arsenal's second key, is a pitch column)
LOOKUP_KEY_COLUMNS = ["game_date", "batter_id", "pitcher_id"]

# In the training rows only to split them (holdout and sampling by game); left out of the
# logged model's inputs, so scoring a hypothetical pitch doesn't need one
TRAINING_ONLY_COLUMNS = ["game_pk"]

# Late and close (the Elias definition): seventh inning or later, with the batting team
# ahead by one, tied, or with the tying run on base, at bat or on deck
LATE_INNING = 7

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
    *STATE_COLUMNS,
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


def request_columns_sql() -> str:
    """SQL select items for the pitch's own columns and the game state, over fact_pitch."""
    columns = {**PITCH_COLUMNS, **STATE_COLUMNS}
    return ",\n  ".join(f"{expr} AS {name}" for name, expr in columns.items())


def labels_sql(fact_pitch: str, where: str, label: str) -> str:
    """SQL for training rows: lookup keys, the pitch and state columns, and the label.

    fe.log_model records every non-label column as a model input unless the training set
    excludes it, so the only extra here is TRAINING_ONLY_COLUMNS (excluded when logging);
    the training job derives the season from game_date itself.
    """
    return f"""
SELECT
  {", ".join([*TRAINING_ONLY_COLUMNS, *LOOKUP_KEY_COLUMNS])},
  {request_columns_sql()},
  {label} AS label
FROM {fact_pitch}
WHERE {where}
"""


def feature_lookups(ml_schema: str) -> list:
    """FeatureLookups for the batter and arsenal feature tables in `ml_schema`."""
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
    ]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features and return the model's input columns (FEATURES), in order.

    `df` has the pitch and state columns plus the looked-up features. Missing lookups (a
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

    # Game state: the raw columns, then what follows from them
    for col in ["balls", "strikes", "outs", "inning", "bat_score_diff", "times_through_order"]:
        out[col] = numeric(col)
    balls, strikes = out["balls"], out["strikes"]
    count_group = pd.Series(
        np.select(
            [strikes == 2, balls > strikes, balls.notna() & strikes.notna()],
            ["two_strikes", "hitter_ahead", "neutral"],
            default=None,
        ),
        index=df.index,
    )
    base_state = df["base_state"].astype("string")
    runners = base_state.str.replace("_", "").str.len().astype("float32")
    no_base = base_state.isna().to_numpy()
    out["runners_on"] = np.where(no_base, np.nan, runners > 0).astype("float32")
    scoring_position = base_state.str.contains("2") | base_state.str.contains("3")
    out["runner_in_scoring_position"] = np.where(
        no_base, np.nan, scoring_position.fillna(False)
    ).astype("float32")
    diff = out["bat_score_diff"]
    late_and_close = (out["inning"] >= LATE_INNING) & (diff <= 1) & (-diff <= runners + 2)
    known = out["inning"].notna() & diff.notna() & ~no_base
    out["is_late_and_close"] = np.where(known, late_and_close, np.nan).astype("float32")
    out["is_first_pitch_of_pa"] = np.where(
        balls.notna() & strikes.notna(), (balls == 0) & (strikes == 0), np.nan
    ).astype("float32")
    same_side = df["pitcher_throws"] == df["batter_stands"]
    known = (df["pitcher_throws"].notna() & df["batter_stands"].notna()).to_numpy()
    out["is_platoon_same_side"] = np.where(known, same_side, np.nan).astype("float32")

    for col in [*BATTER_FEATURES, *ARSENAL_FEATURES]:
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
        df["pitch_group"].astype("string") + "_" + region + "_" + count_group.astype("string"),
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

    categorical = {col: count_group if col == "count_group" else df[col] for col in CATEGORIES}
    for col, levels in CATEGORIES.items():
        # Values outside the known levels (e.g. a new pitch type code) become missing
        values = categorical[col]
        known = values.where(values.isin(levels))
        out[col] = pd.Categorical(known, categories=levels)

    return pd.DataFrame(out, index=df.index)[FEATURES]
