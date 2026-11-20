"""Tests for bimanual.conditions (port of tuningAnalysis.m labeling helpers).

Expected values were worked out by hand from the MATLAB source. They are
double-checked against real MATLAB/Octave output in step 0.5.
"""

import numpy as np
import pytest

from bimanual import conditions as C


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (1, 1, 45.0),
        (-1, 1, 135.0),
        (-1, -1, 225.0),
        (1, -1, 315.0),
        (0, 1, 90.0),
        (-1, 0, 180.0),
        (0, -1, 270.0),
        (0, 0, 0.0),  # MATLAB returns 0 for the zero vector
    ],
)
def test_angle_from_x_axis(x, y, expected):
    assert C.angle_from_x_axis(x, y) == pytest.approx(expected)


def test_every_integer_angle_has_exactly_one_wedge():
    counts = np.zeros(9, int)
    for a in range(360):
        hits = [w for w, lo in enumerate(C._WEDGE_LO, 1) if (a - lo) % 360 < 45]
        assert len(hits) == 1
        counts[hits[0]] += 1
    assert (counts[1:] == 45).all()


@pytest.mark.parametrize(
    "deg, wedge",
    [
        # wedge boundaries from targetMapping (tuningAnalysis.m:30)
        (156, 8), (157, 1), (201, 1), (202, 2), (246, 2), (247, 3),
        (291, 3), (292, 4), (336, 4), (337, 5), (359, 5), (0, 5),
        (21, 5), (22, 6), (66, 6), (67, 7), (111, 7), (112, 8),
    ],
)
def test_wedge_boundaries(deg, wedge):
    assert C.wedge_of_angle(deg) == wedge


def test_wedge_centers_match_ids():
    for wedge_id, center in enumerate(C.WEDGE_CENTERS_DEG, 1):
        assert C.wedge_of_angle(int(center)) == wedge_id


def test_which_target_floor_and_axis_quirks():
    # 21.9 deg floors to 21 -> wedge 5 (0 deg); 22.0 deg -> wedge 6 (45 deg)
    a = np.radians(21.9)
    assert C.which_target(np.array([np.cos(a), np.sin(a)])) == 5
    a = np.radians(22.5)
    assert C.which_target(np.array([np.cos(a), np.sin(a)])) == 6
    # exactly on an axis -> NaN (MATLAB quirk kept)
    assert np.isnan(C.which_target(np.array([0.0, 0.5])))
    assert np.isnan(C.which_target(np.array([0.5, 0.0])))
    # just below the +x axis: ~359.99 deg -> floor 359 -> wedge 5
    assert C.which_target(np.array([1.0, -1e-6])) == 5


def test_trial_conditions_uses_1_based_go_cue_and_idle_hand():
    n_bins = 10
    Tp = np.zeros((4, n_bins))
    Cp = np.zeros((4, n_bins))
    # At bin index 4 (MATLAB index 5): RH target up-left (135 deg), LH down-right (315 deg).
    Tp[:, 4] = [-1, 1, 1, -1]
    out = C.trial_conditions(Tp, Cp, go_cue=[5, 5, 5], move_type=[1, 2, 3])
    np.testing.assert_array_equal(out[0], [1, 8, 9])  # uniR: LH forced to 9
    np.testing.assert_array_equal(out[1], [2, 9, 4])  # uniL: RH forced to 9
    np.testing.assert_array_equal(out[2], [3, 8, 4])  # bimanual
