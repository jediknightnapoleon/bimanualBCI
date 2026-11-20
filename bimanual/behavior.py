"""Closed-loop behavioural metrics: time to acquisition, angular error, speed.

Reproduces
----------
* Fig 5a-c: time to acquisition (TTA) and angular error of online two-cursor
  control, re-scored from the logged cursor/target traces.
* Fig 4e:   cursor speed profiles (raw-trained vs augmented RNN).

Task rules (Methods / Supp Movie 1)
-----------------------------------
* A trial succeeds when BOTH cursors are inside their targets at the same
  time, without a break, for 500 ms ("dwell"). Leaving a target resets the
  dwell timer.
* Trials time out after 10 s and then count as failures.
* Angular error is the average difference between the movement direction and
  the target direction from 300 to 500 ms after the go cue, which captures
  the initial ballistic movement.

Conventions
-----------
* Positions are arrays of shape (4, n_bins) with rows [RH x, RH y, LH x, LH y],
  the same layout as ``Cp``/``Tp`` in the data files.
* One bin = 20 ms (``BIN_S``).
* Bin indices passed to these functions are **0-based** (Python). The data
  files store 1-based MATLAB indices; ``bimanual.io`` converts them on load.

Assumptions (also listed in docs/methods_notes.md)
--------------------------------------------------
* ASSUMPTION: TTA is the time from the go cue to the END of the successful
  500-ms dwell, the moment the task registers the acquisition. Pass
  ``include_dwell=False`` to get the time the final dwell started instead.
* ASSUMPTION: the target direction for the angular error is fixed at the go
  cue (target minus cursor at the go bin).
* ASSUMPTION: movement direction comes from the per-bin change in cursor
  position. Bins where the cursor doesn't move are skipped.
"""

from __future__ import annotations

import numpy as np

BIN_S = 0.020  # 20-ms bins
DWELL_S = 0.5  # simultaneous dwell needed for success
TIMEOUT_S = 10.0  # maximum trial length
ANG_WINDOW_S = (0.3, 0.5)  # angular-error window after go cue

#: Row slices of the (4, n_bins) position arrays for each hand.
HAND_ROWS = {"R": slice(0, 2), "L": slice(2, 4)}


def in_target(cursor: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    """Boolean per bin: is the cursor inside the target circle?

    Args:
        cursor, target: shape (2, n_bins), positions of one hand.
        radius: target radius, in the same units as the positions.
    """
    return np.hypot(*(cursor - target)) <= radius


def time_to_acquisition(
    Cp: np.ndarray,
    Tp: np.ndarray,
    go_bin: int,
    radius_R: float,
    radius_L: float | None = None,
    end_bin: int | None = None,
    include_dwell: bool = True,
) -> float:
    """Seconds from the go cue until both cursors finish a 500-ms dwell.

    Args:
        Cp, Tp: cursor and target positions, shape (4, n_bins).
        go_bin: 0-based bin of the go cue.
        radius_R, radius_L: target radii for each hand (``radius_L`` defaults to
            ``radius_R``).
        end_bin: optional 0-based last bin of the trial; the search also never
            runs past the 10-s timeout.
        include_dwell: if True (default), return the time at which the dwell
            completes; if False, the time the successful dwell started.

    Returns:
        TTA in seconds, or ``np.nan`` if the trial failed (no full dwell
        before the timeout).
    """
    radius_L = radius_R if radius_L is None else radius_L
    n_timeout = int(round(TIMEOUT_S / BIN_S))
    n_dwell = int(round(DWELL_S / BIN_S))
    stop = go_bin + n_timeout
    if end_bin is not None:
        stop = min(stop, end_bin + 1)
    stop = min(stop, Cp.shape[1])

    seg = slice(go_bin, stop)
    both_in = in_target(Cp[HAND_ROWS["R"], seg], Tp[HAND_ROWS["R"], seg], radius_R) & in_target(
        Cp[HAND_ROWS["L"], seg], Tp[HAND_ROWS["L"], seg], radius_L
    )

    # Walk forward counting consecutive "both inside" bins; any exit resets the count.
    run = 0
    for k, inside in enumerate(both_in):
        run = run + 1 if inside else 0
        if run == n_dwell:
            done_k = k + 1  # bins elapsed since the go cue when the dwell completes
            start_k = done_k - n_dwell
            return (done_k if include_dwell else start_k) * BIN_S
    return np.nan


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to the range (-180, 180]."""
    return -((-np.asarray(a) + 180.0) % 360.0 - 180.0)


def angular_error(
    Cp: np.ndarray, Tp: np.ndarray, go_bin: int, hand: str, window_s=ANG_WINDOW_S
) -> float:
    """Mean absolute angle (degrees) between cursor movement and target direction.

    Uses the bins whose START lies within ``window_s`` after the go cue
    (300-500 ms gives 10 bins of 20 ms). The movement in bin t is
    ``Cp[:, t+1] - Cp[:, t]``.

    Args:
        Cp, Tp: shape (4, n_bins).
        go_bin: 0-based go-cue bin.
        hand: "R" or "L".

    Returns:
        Angular error in degrees within [0, 180], or NaN if the cursor never
        moved in the window or the target is at the cursor.
    """
    rows = HAND_ROWS[hand]
    to_target = Tp[rows, go_bin] - Cp[rows, go_bin]
    if not np.any(to_target):
        return np.nan
    target_ang = np.degrees(np.arctan2(to_target[1], to_target[0]))

    b0 = go_bin + int(round(window_s[0] / BIN_S))
    b1 = go_bin + int(round(window_s[1] / BIN_S))  # exclusive
    disp = np.diff(Cp[rows, b0 : b1 + 1], axis=1)  # (2, b1 - b0)
    moving = np.hypot(*disp) > 0
    if not moving.any():
        return np.nan
    move_ang = np.degrees(np.arctan2(disp[1, moving], disp[0, moving]))
    return float(np.mean(np.abs(_wrap_deg(move_ang - target_ang))))


def speed_profile(
    Cp: np.ndarray, go_bin: int, hand: str, n_bins: int, pre_bins: int = 0
) -> np.ndarray:
    """Cursor speed per bin around the go cue, in position-units per second (Fig 4e).

    Returns shape (pre_bins + n_bins,). Bin t's speed is
    ``|Cp[:, t+1] - Cp[:, t]| / BIN_S``, for t from ``go_bin - pre_bins`` to
    ``go_bin + n_bins - 1``. Bins past the end of the recording are NaN.
    """
    rows = HAND_ROWS[hand]
    out = np.full(pre_bins + n_bins, np.nan)
    for i, t in enumerate(range(go_bin - pre_bins, go_bin + n_bins)):
        if 0 <= t and t + 1 < Cp.shape[1]:
            out[i] = np.hypot(*(Cp[rows, t + 1] - Cp[rows, t])) / BIN_S
    return out
