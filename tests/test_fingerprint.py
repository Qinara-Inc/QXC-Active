"""The report subject block: stable across runs, machines, hash seeds."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from qxc_active import Circuit, Distribution, Gate, active_reduce
from qxc_active.fingerprint import (
    circuit_fingerprint,
    observable_fingerprint,
)
from qxc_active.paulis import Observable, PauliTerm


def _circ() -> Circuit:
    c = Circuit(3).add("h", 0).add("cx", (0, 1)).add("rz", 2, 0.375)
    c.append(
        Gate(
            "custom", (1, 2), (),
            matrix=np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=complex,
            ),
        )
    )
    return c


def test_identical_inputs_identical_hashes():
    assert circuit_fingerprint(_circ()) == circuit_fingerprint(_circ())


def test_any_change_changes_the_hash():
    base = circuit_fingerprint(_circ())
    c2 = _circ()
    c2.add("x", 0)
    assert circuit_fingerprint(c2) != base
    c3 = Circuit(3).add("h", 0).add("cx", (0, 1)).add("rz", 2, 0.3750001)
    assert circuit_fingerprint(c3) != circuit_fingerprint(
        Circuit(3).add("h", 0).add("cx", (0, 1)).add("rz", 2, 0.375)
    )


def test_observable_hash_ignores_name_and_term_order():
    a = Observable(
        [(0.5, PauliTerm.parse("Z0")), (0.25, PauliTerm.parse("X1"))],
        name="alice",
    )
    b = Observable(
        [(0.25, PauliTerm.parse("X1")), (0.5, PauliTerm.parse("Z0"))],
        name="bob",
    )
    assert observable_fingerprint(a) == observable_fingerprint(b)
    c = Observable([(0.5, PauliTerm.parse("Z0"))])
    assert observable_fingerprint(a) != observable_fingerprint(c)
    d = Distribution([0, 2])
    assert observable_fingerprint(d) == observable_fingerprint(
        Distribution([2, 0])
    )
    assert observable_fingerprint(d) != observable_fingerprint(
        Distribution([0, 1])
    )


def test_report_carries_subject_and_schema_version():
    _, rep = active_reduce(_circ(), ["Z0", Distribution([0, 1])])
    blob = rep.to_dict()
    assert blob["schema_version"]
    assert blob["schema_version"] != blob["version"]
    subj = blob["subject"]
    assert subj["circuit_sha256"] == circuit_fingerprint(_circ())
    assert len(subj["observables_sha256"]) == 2
    assert subj["options"]["epsilon"] == 0.0
    assert subj["options"]["max_terms"] == 20_000


def test_subject_options_reflect_resolved_options():
    _, rep = active_reduce(
        _circ(), ["Z0"], epsilon=0.05, max_terms=64, compact=False
    )
    opts = rep.subject["options"]
    assert opts["epsilon"] == 0.05
    assert opts["max_terms"] == 64
    assert opts["compact"] is False


_SEED_PROBE = """
import json, sys
sys.path.insert(0, {src!r})
from qxc_active import Circuit
from qxc_active.fingerprint import circuit_fingerprint, observable_fingerprint
from qxc_active.paulis import Observable, PauliTerm
c = Circuit(3).add("h", 0).add("cx", (0, 1)).add("rz", 2, 0.375)
o = Observable([(0.5, PauliTerm.parse("Z0")), (-1.5, PauliTerm.parse("X1*Y2"))])
print(json.dumps([circuit_fingerprint(c), observable_fingerprint(o)]))
"""


def test_hashes_independent_of_pythonhashseed(tmp_path):
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    outs = []
    for seed in ("0", "1", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run(
            [sys.executable, "-c", _SEED_PROBE.format(src=src)],
            capture_output=True, text=True, env=env, check=True,
        )
        outs.append(json.loads(r.stdout))
    assert outs[0] == outs[1] == outs[2]
