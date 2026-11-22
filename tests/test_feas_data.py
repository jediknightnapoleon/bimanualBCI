"""Phase 10: the day/blocks generalization of fig4_rnn.Data (needs the Dryad .mat files)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bimanual import sessions as S  # noqa: E402

pytestmark = pytest.mark.skipif(not S.get(1750).path.exists(), reason="Dryad data not present")


def test_day1750_split_has_two_hand_test_trials():
    import ext_feasibility as E
    import fig4_rnn as F
    blocks = E.ol_split(1750)
    assert blocks[2] == (27, 28) and blocks[1] == (25, 26) and blocks[0][0] == 12
    d = F.Data("clip", day=1750, blocks=blocks)
    for split, bl in zip(("train", "val", "test"), blocks):
        assert len(d.seg[split]) == len(bl)
        for (a, b), blk in zip(d.seg[split], bl):
            assert np.all(d.block[a:b] == blk)
    mt = np.concatenate([d.move_type_bin[a:b] for a, b in d.seg["test"]])
    assert (mt == 3).sum() > 5000  # two-hand trial bins present in the test blocks
