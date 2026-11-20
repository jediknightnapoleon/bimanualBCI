"""Step 0.5: check the Python port against MATLAB/Octave output of tuningAnalysis.m.

Purpose
-------
``matlab_reference/export_tuning_reference.m`` runs the authors' own code,
unchanged, and saves its tuning curves. This script recomputes them with
``bimanual.tuning`` on the same session and compares:

* trial condition labels [moveType, RH wedge, LH wedge]: must match exactly;
* the four tuning-curve matrices (192 x 8): must match to about 1e-10.

Every reference file found is checked: ``tuning_ref_octave.mat`` (Windows,
Octave) and ``tuning_ref_matlab.mat`` (the user's Mac, once it exists).

Inputs : results/reference/tuning_ref_*.mat, Data/t5_06_02_2021.mat
Outputs: results/reference/compare_reference.json; prints PASS/FAIL
Run    : pixi run compare-reference      (about 10 s)
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.io as sio

from bimanual.cache import RESULTS_DIR, atomic_write_json
from bimanual.io import load_session
from bimanual.tuning import tuning_curves

TOL = 1e-10
REF_DIR = RESULTS_DIR / "reference"


def main() -> int:
    refs = sorted(REF_DIR.glob("tuning_ref_*.mat"))
    if not refs:
        print(f"No reference files in {REF_DIR}; run matlab_reference/export_tuning_reference.m first.")
        return 1
    py = tuning_curves(load_session(1750))
    report, ok_all = {}, True
    for ref_path in refs:
        ref = sio.loadmat(ref_path)
        engine = ref_path.stem.replace("tuning_ref_", "")
        res = {}
        # Labels must be identical (NaN == NaN counts as equal).
        res["conditions_equal"] = bool(np.array_equal(ref["conditions"], py.conditions, equal_nan=True))
        for name in ("uniR", "uniL", "biR", "biL"):
            r, p = ref[f"{name}_tc"], getattr(py, name)
            res[f"{name}_max_abs_diff"] = float(np.nanmax(np.abs(r - p))) if r.shape == p.shape else None
            res[f"{name}_nan_pattern_equal"] = bool(np.array_equal(np.isnan(r), np.isnan(p)))
        ok = res["conditions_equal"] and all(
            res[f"{n}_max_abs_diff"] is not None and res[f"{n}_max_abs_diff"] < TOL and res[f"{n}_nan_pattern_equal"]
            for n in ("uniR", "uniL", "biR", "biL")
        )
        res["pass"] = ok
        ok_all &= ok
        report[engine] = res
        print(f"[{'PASS' if ok else 'FAIL'}] {engine}: " + ", ".join(f"{k}={v}" for k, v in res.items()))
    atomic_write_json(REF_DIR / "compare_reference.json", report)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
