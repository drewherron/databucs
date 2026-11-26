import numpy as np
import pandas as pd
import pytest

from databucs.models import features
from databucs.models.outcomes import MODELS
from databucs.models.run_values import RESULTS
from databucs.recommender import options
from databucs.recommender.serving import RequestError, Snapshot, recommend

PITCHER, BATTER = 1, 2


class StubModel:
    """Swings at strikes, takes balls; a called pitch in the zone is a strike."""

    def __init__(self, name, classes):
        self.name, self.classes = name, classes

    def predict(self, context, df):
        in_zone = df["is_in_zone"].astype(float).to_numpy()
        n = len(df)
        if self.name == "swing":
            p = {"take": 1 - 0.6 * in_zone, "swing": 0.6 * in_zone}
        elif self.name == "take_outcome":
            p = {"ball": 1 - in_zone, "called_strike": in_zone, "hit_by_pitch": np.zeros(n)}
        elif self.name == "whiff":
            p = {"contact": np.full(n, 0.8), "whiff": np.full(n, 0.2)}
        elif self.name == "contact_outcome":
            p = {"foul": np.full(n, 0.5), "in_play": np.full(n, 0.5)}
        else:
            p = {c: np.full(n, 0.2) for c in self.classes}
        return pd.DataFrame({f"p_{c}": p[c] for c in self.classes})


MODELS_STUB = {m.name: StubModel(m.name, list(m.classes)) for m in MODELS}


def snapshot(usage_zone_value=9) -> Snapshot:
    arsenal_row = {name: 1.0 for name in features.ARSENAL_FEATURES}
    arsenal_row.update(
        pitcher_id=PITCHER, pitch_type="FF", pitch_group="fastball", pitch_name="4-Seam Fastball",
        velocity=95.0, spin_rate=2300.0, horizontal_break=8.0, induced_vertical_break=16.0,
        release_side=2.0, release_height=6.0, extension=6.5, arm_angle=40.0,
        usage_rate=0.5, pitches_thrown=900, pitcher_total_pitches=300,
    )  # fmt: skip
    profile = {name: 0.3 for name in [*features.BATTER_FEATURES, *features.BATTER_CELL_COUNTS]}
    land = options.landing_points()[0].land  # far outside: a ball
    rv = {f"rv_{r}": 0.0 for r in RESULTS}
    rv.update(rv_ball=0.05, rv_called_strike=-0.05, rv_whiff=-0.05, rv_single=0.4)
    tables = {
        "batters": pd.DataFrame(
            [{"batter_id": BATTER, "batter_name": "B", "bats": "L", "sz_top": 3.4, "sz_bot": 1.6}]
        ),
        "sides": pd.DataFrame(columns=["batter_id", "pitcher_throws", "batter_stands"]),
        "pitchers": pd.DataFrame(
            [{"pitcher_id": PITCHER, "pitcher_name": "P", "pitcher_throws": "R"}]
        ),
        "arsenal": pd.DataFrame([arsenal_row]),
        "batter_features": pd.DataFrame([{"batter_id": BATTER, **profile}]),
        "usage_land": pd.DataFrame(
            [
                {
                    "pitcher_id": PITCHER,
                    "balls": 1,
                    "strikes": 1,
                    "same_side": False,
                    "pitch_type": "FF",
                    "land": land,
                    "usage": 1.0,
                    "usual_by_exact_count": True,
                }
            ]
        ),  # fmt: skip
        "usage_zone": pd.DataFrame(
            [
                {
                    "pitcher_id": PITCHER,
                    "balls": 1,
                    "strikes": 1,
                    "same_side": False,
                    "pitch_type": "FF",
                    "zone": usage_zone_value,
                    "usage": 1.0,
                    "usual_by_exact_count": True,
                }
            ]
        ),  # fmt: skip
        "run_values": pd.DataFrame(
            [{"balls": 1, "strikes": 1, "outs": 1, "base_state": "___", **rv}]
        ),
    }
    return Snapshot("2026-09-25", tables)


REQUEST = {"pitcher_id": PITCHER, "batter_id": BATTER, "balls": 1, "strikes": 1, "outs": 1,
           "base_state": "___"}  # fmt: skip


def test_recommend_scores_every_zone_and_ranks_them():
    out = recommend(snapshot(), MODELS_STUB, REQUEST)
    assert len(out) == len(options.ZONES)  # one pitch type
    assert out["option_rank"].min() == 1 and out["option_rank"].is_monotonic_increasing
    assert out["batter_stands"].iloc[0] == "L"  # listed side, no observed side
    assert out["pitcher_limited_data"].iloc[0]  # 300 pitches
    probs = out[[f"p_{r}" for r in RESULTS]].sum(axis=1)
    assert probs.to_numpy() == pytest.approx(1.0)
    # Strikes are good for the pitcher here, so the middle beats a chase pitch
    value = out.set_index("zone")["expected_run_value"]
    assert value[5] < value[14]
    assert (out["expected_run_value_low"] <= out["expected_run_value"]).all()


def test_usual_mix_and_tiers():
    out = recommend(snapshot(), MODELS_STUB, REQUEST)
    # His usual pitch lands far outside (a ball, +0.05), so in-zone aims are strong
    assert out["usual_expected_run_value"].iloc[0] == pytest.approx(0.05)
    middle = out.set_index("zone").loc[5]
    assert middle["tier"] == "strong" and middle["vs_usual"] < 0
    assert out.set_index("zone").loc[9, "usual_share"] == 1.0
    assert out.set_index("zone").loc[5, "location"] == "middle"


def test_a_wider_miss_pulls_zones_toward_each_other():
    tight = recommend(snapshot(), MODELS_STUB, REQUEST, miss_sd_ft=0.2)
    wide = recommend(snapshot(), MODELS_STUB, REQUEST, miss_sd_ft=1.0)

    def spread(out):
        return out["expected_run_value"].max() - out["expected_run_value"].min()

    assert spread(wide) < spread(tight)


@pytest.mark.parametrize(
    "change",
    [{"pitcher_id": 99}, {"batter_id": 99}, {"balls": 4}, {"base_state": "2__"}],
)
def test_bad_requests_raise(change):
    with pytest.raises(RequestError):
        recommend(snapshot(), MODELS_STUB, {**REQUEST, **change})
