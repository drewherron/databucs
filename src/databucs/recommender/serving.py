"""The recommender for one request, in pandas: the same scoring as the batch job
(jobs/recommend_pitches.py), for real-time "what if" questions on a serving endpoint.

Everything the batch job looks up in Spark comes from a Snapshot instead: dated copies of
the batter profiles, strike zones and sides, pitcher arsenals, each pitcher's usual mix and
the run values (built by jobs/register_recommender.py with the batch job's own queries).

For one pitcher, batter and situation:
1. every pitch type in his arsenal at every landing point (options.landing_points), scored
   by the five outcome models and turned into expected run value
2. each zone's nine aim points valued over where an aimed pitch lands (options.aim_weights)
3. his usual mix valued at his real landing spots, the tier and the labels

The registration job checks that this gives the same numbers as the batch table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from databucs.models import features
from databucs.models.run_values import RESULTS, result_probabilities
from databucs.recommender import inputs, options

# The hypothetical pitch's columns that come straight from the arsenal row (see
# inputs.HYPOTHETICAL_PITCH; the rest depend on the matchup and the landing point)
ARSENAL_PITCH_COLUMNS = {
    "pitch_type": "pitch_type",
    "pitch_group": "pitch_group",
    "pitch_velocity": "velocity",
    "pitch_spin_rate": "spin_rate",
    "pitch_horizontal_break": "horizontal_break",
    "pitch_induced_vertical_break": "induced_vertical_break",
    "pitch_release_side": "release_side",
    "pitch_release_height": "release_height",
    "pitch_extension": "extension",
    "pitch_arm_angle": "arm_angle",
}

# Situation defaults, as in the batch scenarios
DEFAULT_STATE = {"inning": 5, "bat_score_diff": 0, "times_through_order": 2}

SNAPSHOT_TABLES = [
    "batters",  # batter_id, batter_name, bats, sz_top, sz_bot
    "sides",  # batter_id, pitcher_throws, batter_stands
    "pitchers",  # pitcher_id, pitcher_name, pitcher_throws
    "arsenal",  # pitcher arsenal rows (inputs.arsenal_sql)
    "batter_features",  # batter_id + the batter profile features
    "usage_land",  # inputs.usage_sql(..., "land") for every count
    "usage_zone",  # inputs.usage_sql(..., "zone") for every count
    "run_values",  # balls, strikes, outs, base_state, rv_<result>
]


class RequestError(ValueError):
    """A request the snapshot can't answer (unknown player, invalid situation)."""


@dataclass
class Snapshot:
    as_of_date: str
    tables: dict[str, pd.DataFrame]

    def __post_init__(self):
        t = self.tables
        self.batters = t["batters"].set_index("batter_id")
        self.pitchers = t["pitchers"].set_index("pitcher_id")
        self.batter_features = t["batter_features"].set_index("batter_id")
        self.sides = t["sides"].set_index(["batter_id", "pitcher_throws"])["batter_stands"]
        self.arsenal = {k: g for k, g in t["arsenal"].groupby("pitcher_id")}
        self.usage_land = {
            k: g
            for k, g in t["usage_land"].groupby(["pitcher_id", "balls", "strikes", "same_side"])
        }
        self.usage_zone = {
            k: g
            for k, g in t["usage_zone"].groupby(["pitcher_id", "balls", "strikes", "same_side"])
        }
        self.run_values = t["run_values"].set_index(["balls", "strikes", "outs", "base_state"])


@lru_cache(maxsize=256)
def _aim_matrix(zone_height_ft: float, sd_ft: float) -> np.ndarray:
    """(aim points x landing points) weights, rows in options.zone_points() order."""
    aims = {(a.zone, a.point): i for i, a in enumerate(options.zone_points())}
    matrix = np.zeros((len(aims), len(options.landing_points())))
    for zone, point, land, weight in options.aim_weights(zone_height_ft, sd_ft):
        matrix[aims[(zone, point)], land] = weight
    return matrix


def _validate(request: dict) -> None:
    if not (0 <= request["balls"] <= 3 and 0 <= request["strikes"] <= 2):
        raise RequestError("balls must be 0-3 and strikes 0-2")
    if not 0 <= request["outs"] <= 2:
        raise RequestError("outs must be 0-2")
    if not re.fullmatch(r"[_1][_2][_3]", str(request["base_state"])):
        raise RequestError("base_state looks like '___', '1__', '_2_', '1_3' or '123'")


def recommend(
    snapshot: Snapshot,
    models: dict,
    request: dict,
    miss_sd_ft: float = options.MISS_SD_FT,
) -> pd.DataFrame:
    """Every option (pitch type x zone) for one request, ranked; same columns and meaning
    as the batch table. `models` maps outcome model name to an OutcomePyfunc."""
    request = {**DEFAULT_STATE, **{k: v for k, v in request.items() if v is not None}}
    _validate(request)
    pitcher_id, batter_id = int(request["pitcher_id"]), int(request["batter_id"])
    if pitcher_id not in snapshot.pitchers.index or pitcher_id not in snapshot.arsenal:
        raise RequestError(f"no arsenal for pitcher {pitcher_id} as of {snapshot.as_of_date}")
    if batter_id not in snapshot.batters.index:
        raise RequestError(f"no data for batter {batter_id} as of {snapshot.as_of_date}")
    pitcher = snapshot.pitchers.loc[pitcher_id]
    batter = snapshot.batters.loc[batter_id]
    throws = pitcher["pitcher_throws"]
    stands = snapshot.sides.get((batter_id, throws))
    if stands is None or pd.isna(stands):
        stands = ("L" if throws == "R" else "R") if batter["bats"] == "S" else batter["bats"]
    sz_top, sz_bot = float(batter["sz_top"]), float(batter["sz_bot"])

    # 1. Every pitch type at every landing point
    arsenal = snapshot.arsenal[pitcher_id].reset_index(drop=True)
    landing = pd.DataFrame([vars(p) for p in options.landing_points()])
    rows = arsenal.merge(landing, how="cross")
    profile_columns = [*features.BATTER_FEATURES, *features.BATTER_CELL_COUNTS]
    profile = (
        snapshot.batter_features.loc[batter_id]
        if batter_id in snapshot.batter_features.index
        else pd.Series(np.nan, index=profile_columns)  # no recent profile: unknown
    )
    columns = {name: rows[column] for name, column in ARSENAL_PITCH_COLUMNS.items()}
    columns.update(
        pitcher_throws=throws,
        batter_stands=stands,
        is_in_zone=rows["zone"] <= 9,
        plate_x=rows["plate_x"],
        plate_z=sz_bot + rows["height"] * (sz_top - sz_bot),
        sz_top=sz_top,
        sz_bot=sz_bot,
    )
    columns.update({name: request[name] for name in features.STATE_COLUMNS})
    columns.update({name: rows[name] for name in features.ARSENAL_FEATURES})
    columns.update({name: profile[name] for name in profile_columns})
    df = pd.DataFrame(columns, index=rows.index)

    predictions = {name: model.predict(None, df) for name, model in models.items()}
    probabilities = result_probabilities(predictions)
    state = (request["balls"], request["strikes"], request["outs"], request["base_state"])
    values = snapshot.run_values.loc[state]
    landing_value = sum(np.asarray(probabilities[r]) * values[f"rv_{r}"] for r in RESULTS)

    # 2. Aim points over where they land, then zones over their aim points
    n_types, n_land = len(arsenal), len(landing)
    weights = _aim_matrix(sz_top - sz_bot, float(miss_sd_ft))
    aims = pd.DataFrame([vars(a) for a in options.zone_points()])[["zone", "point"]]
    by_type = landing_value.reshape(n_types, n_land)
    probs_by_type = {r: np.asarray(probabilities[r]).reshape(n_types, n_land) for r in RESULTS}
    frames = []
    for i, pitch in arsenal.iterrows():
        aim = aims.copy()
        aim["expected_run_value"] = weights @ by_type[i]
        for r in RESULTS:
            aim[f"p_{r}"] = weights @ probs_by_type[r][i]
        option = aim.groupby("zone").agg(
            expected_run_value=("expected_run_value", "mean"),
            expected_run_value_low=("expected_run_value", "min"),
            expected_run_value_high=("expected_run_value", "max"),
            **{f"p_{r}": (f"p_{r}", "mean") for r in RESULTS},
        )
        option["pitch_type"], option["pitch_name"] = pitch["pitch_type"], pitch["pitch_name"]
        option["pitch_usage_rate"] = pitch["usage_rate"]
        option["pitcher_pitches_thrown"] = pitch["pitches_thrown"]
        frames.append(option.reset_index())
    result = pd.concat(frames, ignore_index=True)

    # 3. His usual mix at his real landing spots, his share of each option, tier, labels
    key = (pitcher_id, request["balls"], request["strikes"], throws == stands)
    usage = snapshot.usage_land.get(key)
    usual = covered = None
    by_exact = None
    if usage is not None:
        type_index = {t: i for i, t in enumerate(arsenal["pitch_type"])}
        usage = usage[usage["pitch_type"].isin(type_index)]
        if len(usage):
            idx = usage["pitch_type"].map(type_index).to_numpy()
            values_at = by_type[idx, usage["land"].to_numpy()]
            covered = float(usage["usage"].sum())
            usual = float((usage["usage"] * values_at).sum() / covered)
            by_exact = bool(usage["usual_by_exact_count"].any())
    zone_usage = snapshot.usage_zone.get(key)
    shares = (
        zone_usage.set_index(["pitch_type", "zone"])["usage"]
        if zone_usage is not None
        else pd.Series(dtype=float)
    )
    result["usual_share"] = [
        float(shares.get((t, z), 0.0))
        for t, z in zip(result["pitch_type"], result["zone"], strict=True)
    ]
    result["usual_expected_run_value"] = usual
    result["usual_share_covered"] = covered
    result["usual_by_exact_count"] = by_exact
    result["vs_usual"] = result["expected_run_value"] - usual if usual is not None else np.nan
    result["tier"] = np.select(
        [
            result["vs_usual"] <= -options.STRONG_MARGIN,
            result["vs_usual"] >= options.AVOID_MARGIN,
        ],
        ["strong", "avoid"],
        default="fine",
    )
    labels = {z: options.zone_label(z, stands) for z in options.ZONES}
    result["location"] = result["zone"].map(lambda z: labels[z].location)
    result["option_rank"] = result["expected_run_value"].rank(method="min").astype(int)
    result = result.sort_values(["option_rank", "pitch_type", "zone"]).reset_index(drop=True)
    result.insert(0, "pitcher_name", pitcher["pitcher_name"])
    result.insert(1, "batter_name", batter["batter_name"])
    result.insert(2, "batter_stands", stands)
    total = arsenal["pitcher_total_pitches"].iloc[0] if "pitcher_total_pitches" in arsenal else None
    result["pitcher_limited_data"] = (
        bool(total < inputs.LIMITED_DATA_PITCHES) if total is not None else None
    )
    result["as_of_date"] = snapshot.as_of_date
    return result
