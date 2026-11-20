"""Trial condition labeling: movement type and target-direction wedge.

Reproduces
----------
The labeling used for the tuning curves of Fig 1b-c. This is a faithful port of
three pieces of ``tuningAnalysis.m`` (upstream repo):

* ``angleFromXaxis``       (tuningAnalysis.m:180-202) -> ``angle_from_x_axis``
* ``whichTarget_bimanual`` (tuningAnalysis.m:154-176) -> ``which_target``
* the condition loop       (tuningAnalysis.m:36-53)   -> ``trial_conditions``

Every quirk of the MATLAB code is kept on purpose, so the Python results can
be checked against MATLAB/Octave output to ~1e-10 (step 0.5). In particular:

* The angle is FLOORED to an integer degree before the wedge lookup.
* A vector with x == 0 OR y == 0 (exactly) gets label NaN. This also catches
  "no target" for the idle hand, but it also drops a target lying exactly on
  an axis. It is kept as-is to match MATLAB.

Wedges
------
The 2D plane is split into 8 wedges of 45 deg. Wedge IDs follow the column
order of ``targetMapping`` in tuningAnalysis.m:30, which matches the x-axis
order of the paper's Fig 1b tuning curves::

    wedge ID :   1    2    3    4    5    6    7    8
    center   : 180  225  270  315    0   45   90  135   (degrees)

ID 9 means "this hand does not move" (unimanual trial of the other hand).

Usage
-----
    from bimanual.conditions import trial_conditions
    cond = trial_conditions(Tp, Cp, go_cue, move_type)   # (n_trials, 3)
"""

from __future__ import annotations

import numpy as np

# Movement type codes as stored in the data (`moveType`), see tuningAnalysis.m:34.
UNI_RIGHT = 1
UNI_LEFT = 2
BIMANUAL = 3
MOVE_TYPE_NAMES = {UNI_RIGHT: "uniR", UNI_LEFT: "uniL", BIMANUAL: "bi"}

#: Label for the idle hand on unimanual trials (tuningAnalysis.m:47-50).
NO_MOVE = 9

#: Wedge lower bounds in integer degrees for wedge IDs 1..8. Each wedge spans 45 integer
#: degrees [lo, lo+44]. Wedge 5 wraps around 0 deg (337..359 and 0..21).
#: Taken from tuningAnalysis.m:30.
_WEDGE_LO = np.array([157, 202, 247, 292, 337, 22, 67, 112])

#: Wedge center angles in degrees, in wedge-ID order (for plotting x-axes).
WEDGE_CENTERS_DEG = np.array([180, 225, 270, 315, 0, 45, 90, 135])


def angle_from_x_axis(x: float, y: float) -> float:
    """Angle of vector [x, y] from the +x axis, in degrees within [0, 360).

    Port of ``angleFromXaxis`` (tuningAnalysis.m:180). It uses acos of the
    normalized dot product with [1, 0], then mirrors it for y < 0, exactly as
    the MATLAB code does (not atan2, to keep the same rounding behaviour).
    Returns 0 for the zero vector.
    """
    if x == 0 and y == 0:
        return 0.0
    cos_theta = x / np.hypot(x, y)  # dot([1,0,0],[x,y,0]) / (|u||v|)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    # MATLAB: quadrants with y >= 0 keep theta, otherwise 360 - theta.
    if y >= 0:
        return float(theta)
    return float(360.0 - theta)


def wedge_of_angle(theta_deg_int: int) -> int:
    """Map an integer angle in degrees (0..359) to a wedge ID 1..8."""
    for wedge_id, lo in enumerate(_WEDGE_LO, start=1):
        if (theta_deg_int - lo) % 360 < 45:
            return wedge_id
    raise ValueError(f"angle {theta_deg_int} not in any wedge")  # unreachable for 0..359


def which_target(vec_xy: np.ndarray) -> float:
    """Wedge ID (1..8) of one 2D direction vector, or NaN.

    Port of ``whichTarget_bimanual`` for a single row (tuningAnalysis.m:154).

    Args:
        vec_xy: shape (2,), e.g. target position minus cursor position for one hand.

    Returns:
        Wedge ID as a float (so NaN can be returned), NaN if x or y is exactly 0.
    """
    x, y = float(vec_xy[0]), float(vec_xy[1])
    if x == 0 or y == 0:
        return np.nan  # MATLAB quirk kept on purpose, see module docstring
    theta = int(np.floor(angle_from_x_axis(x, y)))
    return float(wedge_of_angle(theta))


def trial_conditions(
    Tp: np.ndarray, Cp: np.ndarray, go_cue: np.ndarray, move_type: np.ndarray
) -> np.ndarray:
    """Build the per-trial condition table [moveType, RH wedge, LH wedge].

    Port of tuningAnalysis.m:36-53.

    Args:
        Tp: target positions, shape (4, n_bins); rows = [RH x, RH y, LH x, LH y].
        Cp: cursor positions, shape (4, n_bins); same row layout.
        go_cue: go-cue bin index per trial, shape (n_trials,). These are
            **MATLAB 1-based** indices as stored in the .mat file; they are
            converted to 0-based here.
        move_type: per-trial movement type (1 uniR, 2 uniL, 3 bi), shape (n_trials,).

    Returns:
        Float array of shape (n_trials, 3). Column 0 is moveType, columns 1-2
        are the RH and LH wedge IDs (1..8, 9 = hand not moving, NaN = undefined).
    """
    go_cue = np.asarray(go_cue).ravel().astype(int)
    move_type = np.asarray(move_type).ravel()
    out = np.zeros((go_cue.size, 3))
    for i, (g, mt) in enumerate(zip(go_cue, move_type)):
        pos_err = Tp[:, g - 1] - Cp[:, g - 1]  # 1-based -> 0-based bin index
        rh = which_target(pos_err[0:2])
        lh = which_target(pos_err[2:4])
        if mt == UNI_RIGHT:
            lh = NO_MOVE
        elif mt == UNI_LEFT:
            rh = NO_MOVE
        out[i] = (mt, rh, lh)
    return out


# ----------------------------------------------------------------------------- quadrants
#: Quadrant names, IDs 1..4 = [0, 90), [90, 180), [180, 270), [270, 360) degrees.
QUADRANT_NAMES = ("Q1 0-90", "Q2 90-180", "Q3 180-270", "Q4 270-360")


def quadrant(vec_xy: np.ndarray, min_dist: float = 1.0) -> float:
    """Direction quadrant (1..4) of a 2D cursor->target vector, or NaN.

    ASSUMPTION (register #6): the paper's "4 movement-direction quadrants" are
    taken as the axis-aligned quadrants [0, 90), [90, 180), [180, 270) and
    [270, 360) degrees. The paper doesn't define them, and its targets lie at
    random angles that cover all directions evenly.

    Vectors shorter than ``min_dist`` get NaN: the target is at the cursor, so
    there is no direction. This happens for 1 trial in the whole dataset.
    """
    x, y = float(vec_xy[0]), float(vec_xy[1])
    if np.hypot(x, y) < min_dist:
        return np.nan
    ang = np.degrees(np.arctan2(y, x)) % 360.0
    return float(int(ang // 90) + 1)


def trial_quadrants(Tp: np.ndarray, Cp: np.ndarray, go0: np.ndarray, move_type: np.ndarray) -> np.ndarray:
    """Per-trial [RH quadrant, LH quadrant] at the go cue; NaN for the idle hand.

    Args:
        Tp, Cp: (4, n_bins) target / cursor positions.
        go0: (n_trials,) **0-based** go-cue bins.
        move_type: (n_trials,) 1 uniR, 2 uniL, 3 bimanual.

    Returns:
        (n_trials, 2) float array of quadrant IDs 1..4 or NaN.
    """
    out = np.full((len(go0), 2), np.nan)
    for i, (g, mt) in enumerate(zip(np.asarray(go0, int), move_type)):
        err = Tp[:, g] - Cp[:, g]
        if mt in (UNI_RIGHT, BIMANUAL):
            out[i, 0] = quadrant(err[0:2])
        if mt in (UNI_LEFT, BIMANUAL):
            out[i, 1] = quadrant(err[2:4])
    return out


# ----------------------------------------------------------------------------- regressors
def trial_directions(Tp: np.ndarray, Cp: np.ndarray, go0: np.ndarray, move_type: np.ndarray) -> np.ndarray:
    """Unit cursor->target direction of each hand at the go cue, 0 for the idle hand.

    These are the velocity regressors of the encoding model (Eq. 1-2):
    columns [d_rx, d_ry, d_lx, d_ly].

    ASSUMPTION (register #21): Eq. 2 defines d as target minus cursor
    position. We use its DIRECTION at the go cue (unit length), held constant
    over the analysis window. Target distances in this dataset range from
    about 30 to 750 units, and raw position differences would mix distance
    into a direction-tuning model. Tuning-strength RATIOS and correlations
    don't depend on this scale.

    Args:
        Tp, Cp: (4, n_bins); go0: (n_trials,) 0-based; move_type: (n_trials,).

    Returns:
        (n_trials, 4) array. A vector shorter than 1 unit gives 0 (no direction).
    """
    out = np.zeros((len(go0), 4))
    for i, (g, mt) in enumerate(zip(np.asarray(go0, int), move_type)):
        err = Tp[:, g] - Cp[:, g]
        for cols, moves in ((slice(0, 2), mt in (UNI_RIGHT, BIMANUAL)), (slice(2, 4), mt in (UNI_LEFT, BIMANUAL))):
            n = np.hypot(*err[cols])
            if moves and n >= 1.0:
                out[i, cols] = err[cols] / n
    return out
