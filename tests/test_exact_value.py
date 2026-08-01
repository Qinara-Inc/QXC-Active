"""Empty reduced circuits are classically determined — and say so.

The exact_value contract: present in the measurement block iff the
reduced circuit has zero gates; "exactly" wording iff the certified
bound is exactly zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import expectations
from qxc_active import Circuit, Distribution, active_reduce
from qxc_active.bundle import write_bundle
from qxc_active.paulis import Observable


def _idle_measured_circuit(n=3, busy=2):
    """Gates only on `busy`: any observable elsewhere reduces to empty
    with a certified bound of exactly zero (outside-support removals)."""
    return (
        Circuit(n)
        .add("h", busy)
        .add("rz", busy, 0.3)
        .add("rx", busy, 1.1)
    )


def test_z_observable_exact_plus_one():
    _, rep = active_reduce(_idle_measured_circuit(), ["Z0"])
    assert rep.output_metrics["gates"] == 0
    assert rep.certified_bounds[0] == 0.0
    (entry,) = rep.to_dict()["measurement"]["observables"]
    assert entry["exact_value"] == 1.0
    out = rep.summary()
    assert "Reduced circuit is empty: <obs0> = +1.000000 exactly." in out
    assert "No quantum execution required." in out


def test_xy_factor_gives_zero():
    _, rep = active_reduce(_idle_measured_circuit(), ["X0", "Y0*Z1"])
    m = rep.to_dict()["measurement"]["observables"]
    assert [e["exact_value"] for e in m] == [0.0, 0.0]
    assert "= +0.000000 exactly" in rep.summary()


def test_weighted_sum_arithmetic():
    _, rep = active_reduce(
        _idle_measured_circuit(), ["0.5*Z0 + 2*Z0*Z1"]
    )
    (entry,) = rep.to_dict()["measurement"]["observables"]
    assert entry["exact_value"] == 2.5
    assert "= +2.500000 exactly" in rep.summary()


def test_nonzero_bound_uses_within_wording():
    # rz/cp on the measured qubits: removals charge float dust, so the
    # bound is nonzero and the claim about the ORIGINAL runs through it
    c = Circuit(2).add("rz", 0, 0.5).add("cp", (0, 1), 0.3)
    _, rep = active_reduce(c, ["Z0"])
    assert rep.output_metrics["gates"] == 0
    assert rep.certified_bounds[0] > 0.0
    out = rep.summary()
    (line,) = [
        ln for ln in out.splitlines() if "Reduced circuit is empty" in ln
    ]
    assert "within" in line and "of the original" in line
    assert "exactly" not in line
    (entry,) = rep.to_dict()["measurement"]["observables"]
    assert entry["exact_value"] == 1.0


def test_absent_when_gates_remain():
    c = Circuit(2).add("h", 0).add("cx", (0, 1))
    reduced, rep = active_reduce(c, ["Z0"])
    assert reduced.n_gates > 0
    for entry in rep.to_dict()["measurement"]["observables"]:
        assert "exact_value" not in entry
    assert "Reduced circuit is empty" not in rep.summary()


def test_distribution_all_zeros_probability_one():
    _, rep = active_reduce(
        _idle_measured_circuit(), [Distribution([0, 1])]
    )
    (entry,) = rep.to_dict()["measurement"]["observables"]
    assert entry["exact_value"] == {"outcome": "00", "probability": 1.0}
    out = rep.summary()
    assert "outcome 00" in out and "probability 1" in out
    assert "No quantum execution required." in out


_CONSUMER = r"""
import json, sys
blob = json.loads(open(sys.argv[1], encoding="utf-8").read())
entry = blob["measurement"]["observables"][0]
print(repr(entry["exact_value"]))
"""


def test_end_to_end_bundle_consumer(tmp_path):
    circ = _idle_measured_circuit()
    obs = ["0.5*Z0 + 2*Z0*Z1"]
    reduced, rep = active_reduce(circ, obs)
    write_bundle(tmp_path, reduced, rep)
    r = subprocess.run(
        [sys.executable, "-c", _CONSUMER,
         str(tmp_path / "observables.json")],
        capture_output=True, text=True, check=True,
        env=dict(os.environ, PYTHONPATH=str(
            Path(__file__).resolve().parents[1] / "src"
        )),
    )
    consumer_value = float(r.stdout.strip())
    # independent evaluation: the ORIGINAL circuit with the ORIGINAL
    # observable, brute force
    (truth,) = expectations(circ, [Observable.parse("0.5*Z0 + 2*Z0*Z1")])
    # the certified bound is exactly zero; the residual is the reference
    # simulator's own float noise, not ours
    assert rep.certified_bounds[0] == 0.0
    assert abs(consumer_value - truth) < 1e-12
    assert consumer_value == 2.5  # the exact arithmetic value
