"""The five pitch outcome models and their training labels.

A pitch's result is broken into steps, each with its own model, so each one can be
checked and calibrated on its own:

  swing             all competitive pitches: take or swing
  take_outcome      takes: ball, called strike or hit by pitch
  whiff             swings: contact or whiff
  contact_outcome   contact: foul or in play
  in_play_outcome   balls in play: out, single, double, triple or home run

Multiplying along the path gives the probability of every result, e.g.
P(foul) = P(swing) x (1 - P(whiff)) x P(foul | contact).

Label decisions:
- Competitive pitches leave out pitch-clock violations (no pitch was thrown), pitchouts,
  bunt attempts (a tactical choice, not a reaction to the pitch), catcher's interference,
  and untracked pitches (no type, location or velocity).
- A foul tip counts as a whiff: it is always a strike and an out with two strikes, which
  is a swinging strike's effect, not a foul's.
- Reaching on an error or a fielder's choice is an out: the models predict the quality of
  the batted ball, not the fielding. The run values for the "out" class still include what
  errors and fielder's choices actually cost.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPETITIVE = """
    NOT is_pitch_clock_violation
    AND NOT is_bunt_attempt
    AND description NOT IN ('pitchout', 'swinging_pitchout')
    AND pitch_type IS NOT NULL AND pitch_type <> 'PO'
    AND plate_x IS NOT NULL AND plate_z IS NOT NULL AND release_speed IS NOT NULL
    AND events IS DISTINCT FROM 'catcher_interf'
"""

WHIFF = "(is_whiff OR description = 'foul_tip')"


@dataclass(frozen=True)
class OutcomeModel:
    name: str
    description: str
    where: str  # SQL condition on fact_pitch, on top of COMPETITIVE
    label: str  # SQL expression giving the class index
    classes: tuple[str, ...]

    @property
    def is_binary(self) -> bool:
        return len(self.classes) == 2


MODELS = [
    OutcomeModel(
        name="swing",
        description="Probability the batter swings at a pitch.",
        where="TRUE",
        label="int(is_swing)",
        classes=("take", "swing"),
    ),
    OutcomeModel(
        name="take_outcome",
        description="For a pitch the batter takes: ball, called strike or hit by pitch.",
        where="NOT is_swing",
        label="CASE description WHEN 'called_strike' THEN 1 WHEN 'hit_by_pitch' THEN 2 ELSE 0 END",
        classes=("ball", "called_strike", "hit_by_pitch"),
    ),
    OutcomeModel(
        name="whiff",
        description="For a swing: probability of a whiff (foul tips count as whiffs).",
        where="is_swing",
        label=f"int({WHIFF})",
        classes=("contact", "whiff"),
    ),
    OutcomeModel(
        name="contact_outcome",
        description="For contact (not a whiff or foul tip): probability the ball is in play.",
        where=f"is_swing AND NOT {WHIFF}",
        label="int(is_in_play)",
        classes=("foul", "in_play"),
    ),
    OutcomeModel(
        name="in_play_outcome",
        description="For a ball in play: out, single, double, triple or home run.",
        where="is_in_play",
        label=(
            "CASE events WHEN 'single' THEN 1 WHEN 'double' THEN 2 WHEN 'triple' THEN 3"
            " WHEN 'home_run' THEN 4 ELSE 0 END"
        ),
        classes=("out", "single", "double", "triple", "home_run"),
    ),
]

BY_NAME = {m.name: m for m in MODELS}
