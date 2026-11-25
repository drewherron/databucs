"""What the recommender scores: game scenarios and target zones.

Zones follow Statcast's numbering, from the catcher's view (negative plate_x is the
catcher's left):

     11 |          | 12
    ----+----------+----
        | 1  2  3  |
        | 4  5  6  |
        | 7  8  9  |
    ----+----------+----
     13 |          | 14

Zones 1-9 split the strike zone into thirds across the plate and thirds of the batter's
own zone height. Zones 11-14 are everything outside, by quadrant, so each is an L shape
around a corner of the zone.

Aim and miss. A pitcher aims somewhere and the pitch lands around it, so:
- The models score each pitch type on a regular grid of LANDING spots (landing_points).
- Each zone has nine AIM points (zone_points):
  - zones 1-9: a 3 x 3 grid at 1/6, 1/2 and 5/6 of the cell
  - zones 11-14: three points beside the zone, three above or below it and three off the
    corner, a pitch's width or two outside. Real pitches in these zones land a median of
    about 0.1 ft outside the plate and 0.1-0.2 zone heights above or below it.
- An aim point's value is the average over landing spots, weighted by a normal miss
  around it with MISS_SD_FT in each direction (aim_weights). The outermost landing spots
  take the tails, so a wild pitch counts like the nearest spot on the grid's edge, which is
  an obvious ball.
- A zone's value is the average over its nine aim points; their range shows how much it
  matters where in the zone the pitcher aims.

Heights are fractions of the batter's zone (0 = bottom, 1 = top), so the points scale to
each batter; the miss is in feet, so it is a bigger fraction of a short batter's zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PLATE_HALF_WIDTH = 0.83  # ft, Statcast's zone edge: half the plate plus a ball's radius
OUTSIDE_FT = 0.3  # how far outside the plate the "beside" chase points sit
OUTSIDE_HEIGHT = 0.15  # how far above or below the zone the "above/below" points sit

# A normal miss of this size, in feet, in each direction. A rough league-wide value: aimed
# at the middle of the zone, 68% of pitches would land in it; on 3-1, when pitchers most
# want a strike, 63% of real pitches do.
MISS_SD_FT = 0.65

# Landing grid: plate_x from -1.5 to 1.5 ft and height from -0.5 to 1.5 zone heights
LANDING_X0, LANDING_DX, LANDING_NX = -1.5, 0.25, 13
LANDING_H0, LANDING_DH, LANDING_NH = -0.5, 0.125, 17

IN_ZONE = range(1, 10)
CHASE_ZONES = {11: (-1, "high"), 12: (1, "high"), 13: (-1, "low"), 14: (1, "low")}
ZONES = [*IN_ZONE, *CHASE_ZONES]


@dataclass(frozen=True)
class ZonePoint:
    zone: int
    point: int  # 0-8 within the zone
    plate_x: float  # ft, catcher's view
    height: float  # fraction of the batter's zone


def _in_zone_points(zone: int) -> list[ZonePoint]:
    column, row = (zone - 1) % 3, (zone - 1) // 3  # row 0 is the top
    width = 2 * PLATE_HALF_WIDTH / 3
    left = -PLATE_HALF_WIDTH + column * width
    top = 1 - row / 3
    fractions = (1 / 6, 1 / 2, 5 / 6)
    return [
        ZonePoint(zone, 3 * i + j, left + fx * width, top - fy / 3)
        for i, fy in enumerate(fractions)
        for j, fx in enumerate(fractions)
    ]


def _chase_points(zone: int) -> list[ZonePoint]:
    side, end = CHASE_ZONES[zone]
    # Heights measured outward from the zone edge this quadrant touches
    edge, outward = (1.0, 1) if end == "high" else (0.0, -1)
    inside = -outward  # toward the middle of the zone
    beside_x = side * (PLATE_HALF_WIDTH + OUTSIDE_FT)
    beyond_height = edge + outward * OUTSIDE_HEIGHT
    points = [
        # beside the zone, in its near half
        *[(beside_x, edge + inside * f) for f in (0.08, 0.25, 0.42)],
        # above or below the zone, over the near half of the plate
        *[(side * PLATE_HALF_WIDTH * f, beyond_height) for f in (0.17, 0.5, 0.83)],
        # off the corner, moving away
        *[
            (side * (PLATE_HALF_WIDTH + OUTSIDE_FT * k), edge + outward * OUTSIDE_HEIGHT * k)
            for k in (0.5, 1.0, 1.5)
        ],
    ]
    return [ZonePoint(zone, i, x, h) for i, (x, h) in enumerate(points)]


def zone_points() -> list[ZonePoint]:
    """Nine aim points for every zone in ZONES."""
    return [p for z in ZONES for p in (_in_zone_points(z) if z in IN_ZONE else _chase_points(z))]


def statcast_zone(plate_x: float, height: float) -> int:
    """The Statcast zone a location falls in (edges count as in the zone)."""
    if abs(plate_x) <= PLATE_HALF_WIDTH and 0 <= height <= 1:
        third = 2 * PLATE_HALF_WIDTH / 3
        column = min(int((plate_x + PLATE_HALF_WIDTH) // third), 2)
        row = min(int((1 - height) * 3), 2)
        return 1 + 3 * row + column
    high, left = height > 0.5, plate_x < 0
    return {(True, True): 11, (True, False): 12, (False, True): 13, (False, False): 14}[
        (high, left)
    ]


@dataclass(frozen=True)
class LandingPoint:
    land: int  # index into the grid: column * LANDING_NH + row
    plate_x: float
    height: float
    zone: int


def landing_points() -> list[LandingPoint]:
    points = []
    for i in range(LANDING_NX):
        for j in range(LANDING_NH):
            x, h = LANDING_X0 + i * LANDING_DX, LANDING_H0 + j * LANDING_DH
            points.append(LandingPoint(i * LANDING_NH + j, x, h, statcast_zone(x, h)))
    return points


def landing_index_sql(plate_x: str, height: str) -> str:
    """SQL for the nearest landing point's index (clipped to the grid)."""

    def cell(expr: str, start: float, step: float, n: int) -> str:
        return f"least(greatest(int(round(({expr} - {start}) / {step})), 0), {n - 1})"

    column = cell(plate_x, LANDING_X0, LANDING_DX, LANDING_NX)
    row = cell(height, LANDING_H0, LANDING_DH, LANDING_NH)
    return f"({column} * {LANDING_NH} + {row})"


def _cell_masses(target: float, sd: float, start: float, step: float, n: int) -> list[float]:
    """Probability a normal(target, sd) lands nearest each of n grid values; the end cells
    take the tails."""

    def cdf(v: float) -> float:
        return 0.5 * (1 + math.erf((v - target) / (sd * math.sqrt(2))))

    edges = [-math.inf, *[start + (k + 0.5) * step for k in range(n - 1)], math.inf]
    return [cdf(edges[k + 1]) - cdf(edges[k]) for k in range(n)]


def aim_weights(
    zone_height_ft: float, sd_ft: float = MISS_SD_FT, min_weight: float = 1e-4
) -> list[tuple[int, int, int, float]]:
    """(zone, point, land, weight) for every aim point: where a pitch aimed there lands, for
    a batter whose zone is zone_height_ft tall. Tiny weights are dropped and the rest
    rescaled to sum to 1."""
    rows = []
    for aim in zone_points():
        xs = _cell_masses(aim.plate_x, sd_ft, LANDING_X0, LANDING_DX, LANDING_NX)
        hs = _cell_masses(aim.height, sd_ft / zone_height_ft, LANDING_H0, LANDING_DH, LANDING_NH)
        cells = [
            (i * LANDING_NH + j, wx * wh)
            for i, wx in enumerate(xs)
            for j, wh in enumerate(hs)
            if wx * wh >= min_weight
        ]
        total = sum(w for _, w in cells)
        rows += [(aim.zone, aim.point, land, w / total) for land, w in cells]
    return rows


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    balls: int
    strikes: int
    outs: int
    base_state: str  # '1_3' = runners on first and third
    inning: int = 5
    bat_score_diff: int = 0
    times_through_order: int = 2


# The plate-appearance simulator (recommender/simulate.py) starts here and needs every
# count in it, so the bases-empty scenarios cover all twelve.
SIMULATION_OUTS = 1
SIMULATION_BASE_STATE = "___"

# From the report plan, plus the rest of the bases-empty counts for the simulator.
# Situations not named in a scenario (inning, score, times through the order) are a
# mid-game tie the second time through.
SCENARIOS = [
    Scenario("first_pitch", "First pitch, bases empty, one out", 0, 0, 1, "___"),
    Scenario("ahead_0_1", "Pitcher ahead 0-1, bases empty", 0, 1, 1, "___"),
    Scenario("ahead_0_2", "Pitcher ahead 0-2, bases empty", 0, 2, 1, "___"),
    Scenario("behind_1_0", "Pitcher behind 1-0, bases empty", 1, 0, 1, "___"),
    Scenario("even_1_1", "Even 1-1, bases empty", 1, 1, 1, "___"),
    Scenario("ahead_1_2", "Pitcher ahead 1-2, bases empty", 1, 2, 1, "___"),
    Scenario("behind_2_0", "Pitcher behind 2-0, bases empty", 2, 0, 1, "___"),
    Scenario("behind_2_1", "Pitcher behind 2-1, bases empty", 2, 1, 1, "___"),
    Scenario("even_2_2", "Even 2-2, bases empty", 2, 2, 1, "___"),
    Scenario("behind_3_0", "Pitcher behind 3-0, bases empty", 3, 0, 1, "___"),
    Scenario("behind_3_1", "Pitcher behind 3-1, bases empty", 3, 1, 1, "___"),
    Scenario("full_count", "Full count, bases empty", 3, 2, 1, "___"),
    Scenario("need_strikeout", "Runner on third, one out, 1-1", 1, 1, 1, "__3"),
    Scenario("risp_two_outs", "Runner on second, two outs, 1-1", 1, 1, 2, "_2_"),
]


def in_simulation(scenario: Scenario) -> bool:
    """Whether the plate-appearance simulator uses this scenario (its starting state)."""
    return (scenario.outs, scenario.base_state) == (SIMULATION_OUTS, SIMULATION_BASE_STATE)


# Tiers, relative to the pitcher's usual mix in the same situation (runs per pitch). The
# report groups options this way instead of ranking options that are within noise of each
# other. Judgment calls: half a run per 100 pitches better is "strong", a run per 100
# worse is "avoid".
STRONG_MARGIN = 0.005
AVOID_MARGIN = 0.010

GRID_X = ["Off inside", "Inside", "Middle", "Away", "Off away"]
GRID_Y = ["Above", "Up", "Middle", "Low", "Below"]


@dataclass(frozen=True)
class ZoneLabel:
    zone: int
    batter_stands: str
    location: str  # "low and away", "chase up and in", ...
    grid_x: str  # GRID_X: horizontal, from the batter's point of view
    grid_y: str  # GRID_Y


def zone_label(zone: int, batter_stands: str) -> ZoneLabel:
    """Where a zone is from the batter's point of view. Zones are numbered from the
    catcher's view, and a right-handed batter stands on the catcher's left, so for him the
    left column is inside; for a left-handed batter it's away."""
    if zone in IN_ZONE:
        column, row = (zone - 1) % 3, (zone - 1) // 3
        catcher_side = (-1, 0, 1)[column]
    else:
        side, end = CHASE_ZONES[zone]
        catcher_side, row = side, (0 if end == "high" else 2)
    toward_batter = -1 if batter_stands == "R" else 1  # catcher's side the batter is on
    horizontal = (
        "middle" if catcher_side == 0 else ("in" if catcher_side == toward_batter else "away")
    )
    height = ("up", "middle", "low")[row]
    if zone not in IN_ZONE:
        location = f"chase {height} and {horizontal}"
        grid_x = "Off inside" if horizontal == "in" else "Off away"
        grid_y = "Above" if height == "up" else "Below"
    else:
        if height == "middle":
            location = {"middle": "middle", "in": "inside", "away": "away"}[horizontal]
        elif horizontal == "middle":
            location = f"{height}, over the middle"
        else:
            location = f"{height} and {horizontal}"
        grid_x = {"in": "Inside", "middle": "Middle", "away": "Away"}[horizontal]
        grid_y = {"up": "Up", "middle": "Middle", "low": "Low"}[height]
    return ZoneLabel(zone, batter_stands, location, grid_x, grid_y)


def zone_labels() -> list[ZoneLabel]:
    return [zone_label(z, side) for z in ZONES for side in ("L", "R")]
