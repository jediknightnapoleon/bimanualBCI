"""RNN training targets: per-bin cursor velocities and discrete movement context.

Reproduces
----------
Fig 4b and the target definitions used to train the RNN decoder (Methods,
"RNN decoder"; Supp Fig 3):

* Continuous output y_t, shape (4,): x/y velocity for the right cursor, then
  the left cursor.
* Discrete output z_t: one-hot over {no movement, uni-R, uni-L, bimanual}.
* Velocity targets are "saturated, minimum-jerk-like" profiles that start
  after a 200-ms reaction time.

What "saturated minimum-jerk-like" means here
---------------------------------------------
The open-loop cursors moved along minimum-jerk trajectories, whose speed is a
bell curve. A decoder trained on bell-shaped targets learns to produce a
pulse and then stop. The paper instead uses saturated targets, so the
intended speed stays high for most of the movement.

ASSUMPTION: "saturated" = the min-jerk speed bell clipped at a fraction
``sat`` of its peak, then rescaled so the plateau equals 1. This gives a
trapezoid that rises, holds, then falls. ``sat=1`` recovers the plain
min-jerk bell, which Fig 4b uses for comparison. An alternative reading
(rise, then hold until the trial ends) is available with ``hold_to_end=True``.

Context labels and balancing
----------------------------
The RNN's discrete output has 4 classes (``CTX_*``). For balancing the
augmented training snippets, the paper splits "no movement" further into
uni-R delay, uni-L delay, bimanual delay and rest. Those finer labels are
returned as ``stratum`` together with the movement direction quadrant (see
augment.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_S = 0.020
RT_S = 0.200  # reaction time before velocity targets start (Methods)

# Discrete context classes (RNN discrete output z_t)
CTX_NONE, CTX_UNI_R, CTX_UNI_L, CTX_BI = 0, 1, 2, 3
CTX_NAMES = ("none", "uniR", "uniL", "bi")
MOVE_TYPE_TO_CTX = {1: CTX_UNI_R, 2: CTX_UNI_L, 3: CTX_BI}  # data moveType -> context

# Finer "no movement" labels used only for balancing (paper: delay x 3 + rest)
NOMOVE_REST = "rest"
NOMOVE_DELAY = {1: "delay_uniR", 2: "delay_uniL", 3: "delay_bi"}


def minjerk_speed(n_bins: int) -> np.ndarray:
    """Normalized minimum-jerk speed bell over ``n_bins`` bins, with peak 1.

    For s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, the speed is proportional to
    tau^2 (1 - tau)^2. It is sampled at bin centers so it is > 0 everywhere.
    """
    tau = (np.arange(n_bins) + 0.5) / n_bins
    v = tau**2 * (1 - tau) ** 2
    return v / v.max()


def saturated_speed(n_bins: int, sat: float = 0.5, hold_to_end: bool = False) -> np.ndarray:
    """Saturated min-jerk-like speed profile, plateau = 1 (see module docstring).

    Args:
        n_bins: movement duration in bins.
        sat: clip level as a fraction of the min-jerk peak, in (0, 1]. Lower
            values give a flatter, longer plateau. ASSUMPTION: default 0.5.
        hold_to_end: alternative reading. Rise like min-jerk until the clip
            level is reached, then hold at 1 to the end, with no fall.
    """
    if not 0 < sat <= 1:
        raise ValueError("sat must be in (0, 1]")
    v = np.minimum(minjerk_speed(n_bins), sat) / sat
    if hold_to_end:
        first_peak = int(np.argmax(v >= 1.0))
        v[first_peak:] = 1.0
    return v


@dataclass
class TrialSpec:
    """What ``build_targets`` needs to know about one trial (0-based bins).

    Attributes:
        start_bin: first bin of the trial (delay period starts; targets shown).
        go_bin: go-cue bin.
        move_end_bin: bin where the (open-loop) cursor movement ends, exclusive.
            The velocity profile spans go+RT .. move_end_bin.
        end_bin: last bin of the trial, exclusive.
        move_type: 1 = uni-R, 2 = uni-L, 3 = bimanual (data ``moveType``).
        dir_R, dir_L: unit 2-vectors toward each hand's target (ignored for
            the idle hand).
        quadrant_R, quadrant_L: direction quadrant labels (any hashable), or
            None. Used only for balancing in augment.py.
    """

    start_bin: int
    go_bin: int
    move_end_bin: int
    end_bin: int
    move_type: int
    dir_R: np.ndarray
    dir_L: np.ndarray
    quadrant_R: object = None
    quadrant_L: object = None


def build_targets(
    n_bins: int,
    trials: list[TrialSpec],
    sat: float = 0.5,
    hold_to_end: bool = False,
    rt_s: float = RT_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make per-bin training targets for a whole recording.

    Args:
        n_bins: total number of 20-ms bins in the recording.
        trials: trial specs (0-based bins).
        sat, hold_to_end: profile shape (see ``saturated_speed``).
        rt_s: reaction time; velocity starts at go_bin + rt_s / 20 ms.

    Returns:
        vel:     (n_bins, 4) float32, columns [R vx, R vy, L vx, L vy].
        context: (n_bins,) int, one of CTX_*, the RNN discrete target.
        stratum: (n_bins,) object, a fine label for balancing, e.g.
                 "rest", "delay_bi", or ("uniR", qR, None), ("bi", qR, qL).
    """
    vel = np.zeros((n_bins, 4), np.float32)
    context = np.full(n_bins, CTX_NONE, int)
    stratum = np.full(n_bins, NOMOVE_REST, object)
    rt = int(round(rt_s / BIN_S))

    for tr in trials:
        move_start = tr.go_bin + rt
        # The delay period (targets shown, not yet moving) counts as "no
        # movement", tagged by the upcoming trial type for balancing.
        stratum[tr.start_bin : move_start] = NOMOVE_DELAY[tr.move_type]
        n_move = max(tr.move_end_bin - move_start, 0)
        if n_move == 0:
            continue
        prof = saturated_speed(n_move, sat, hold_to_end)
        seg = slice(move_start, move_start + n_move)
        if tr.move_type in (1, 3):
            vel[seg, 0:2] = prof[:, None] * np.asarray(tr.dir_R)[None, :]
        if tr.move_type in (2, 3):
            vel[seg, 2:4] = prof[:, None] * np.asarray(tr.dir_L)[None, :]
        context[seg] = MOVE_TYPE_TO_CTX[tr.move_type]
        name = CTX_NAMES[MOVE_TYPE_TO_CTX[tr.move_type]]
        qR = tr.quadrant_R if tr.move_type in (1, 3) else None
        qL = tr.quadrant_L if tr.move_type in (2, 3) else None
        for t in range(seg.start, seg.stop):
            stratum[t] = (name, qR, qL)
        # Bins after the movement until the trial ends stay "no movement / rest".
    return vel, context, stratum
