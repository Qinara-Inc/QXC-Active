"""The output contract: a consumer holding only reduced.qasm and the
report must be able to measure correctly — nothing else is acceptable
for a tool whose claim is a certified bound."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import expectations
from qxc_active import Circuit, Distribution, active_reduce, dumps_qasm
from qxc_active.errors import InvalidObservableError
from qxc_active.paulis import Observable, PauliTerm


class TestObservableRoundTrip:
    def test_to_dict_from_dict_exact(self):
        obs = Observable.parse("0.5*Z0*Z1 + 0.25*X2 - 1.75*Y0", name="H")
        d = obs.to_dict()
        back = Observable.from_dict(d)
        assert back.name == "H"
        assert sorted(back.terms, key=lambda ct: (ct[1].x, ct[1].z)) == sorted(
            obs.terms, key=lambda ct: (ct[1].x, ct[1].z)
        )

    def test_round_trip_survives_json(self):
        obs = Observable(
            [(1 / 3, PauliTerm.parse("X1*Z4")), (-2.0, PauliTerm.parse("Y0"))]
        )
        back = Observable.from_dict(json.loads(json.dumps(obs.to_dict())))
        # exact float round-trip, not approximate
        assert {t: c for c, t in back.terms} == {t: c for c, t in obs.terms}

    def test_string_field_is_parseable_canonical_form(self):
        obs = Observable.parse("0.5*Z0*Z1 + 0.25*X2")
        reparsed = Observable.parse(obs.to_dict()["string"])
        assert {t: c for c, t in reparsed.terms} == {
            t: c for c, t in obs.terms
        }

    def test_malformed_dict_raises(self):
        with pytest.raises(InvalidObservableError):
            Observable.from_dict({"terms": [{"coefficient": "x"}]})


# The exact defect reproduction from the task: 5 qubits, observable on
# qubits 2 and 3, compaction renumbers them to 0 and 1.
def _defect_case():
    c = (
        Circuit(5)
        .add("h", 2)
        .add("cx", (2, 3))
        .add("rz", 3, 0.4)
        .add("x", 0)
    )
    return c, ["Z2*Z3"]


class TestMeasurementBlock:
    def test_block_contents(self):
        circ, obs = _defect_case()
        reduced, rep = active_reduce(circ, obs)
        blob = rep.to_dict()
        assert blob["schema_version"] == "1.2"
        m = blob["measurement"]
        assert m["qubit_map_pairs"] == [[2, 0], [3, 1]]
        assert m["qubit_map"] == {"2": 0, "3": 1}
        (entry,) = m["observables"]
        assert entry["original"] == "1*Z2*Z3"
        assert entry["reduced"] == "1*Z0*Z1"
        assert entry["certified_bound"] == rep.certified_bounds[0]
        assert entry["reduced_terms"] == [
            {"coefficient": 1.0, "paulis": {"0": "Z", "1": "Z"}}
        ]
        # the old names-only field is untouched (nothing downstream breaks)
        assert blob["observables"] == rep.observable_names

    def test_distribution_entry(self):
        c = Circuit(4).add("h", 1).add("cx", (1, 3))
        _, rep = active_reduce(c, [Distribution([1, 3])])
        (entry,) = rep.to_dict()["measurement"]["observables"]
        assert entry["kind"] == "distribution"
        assert entry["reduced_qubits"] == [0, 1]

    def test_present_on_no_reduction_runs(self):
        c = Circuit(1).add("rx", 0, 1.1).add("ry", 0, 0.9)
        _, rep = active_reduce(c, ["Z0"])
        m = rep.to_dict()["measurement"]
        assert m["observables"][0]["reduced"] == m["observables"][0]["original"]


# Run in a fresh interpreter with NO access to the original observable:
# only the report JSON and the reduced QASM. Reconstructs the remapped
# observable from the report alone and evaluates it.
_CONSUMER = r"""
import json, sys
sys.path.insert(0, {src!r})
from qxc_active import loads_qasm
from qxc_active import sim
from qxc_active.paulis import Observable
report = json.loads(open(sys.argv[1], encoding="utf-8").read())
entry = report["measurement"]["observables"][0]
obs = Observable.from_dict({{"name": entry["name"], "terms": entry["reduced_terms"]}})
circ = loads_qasm(open(sys.argv[2], encoding="utf-8").read())
psi = sim.statevector(circ)
print(repr(sim.expectation_of_state(psi, obs, circ.n_qubits)))
"""


def test_round_trip_consumer_measures_correctly(tmp_path):
    circ, obs_strings = _defect_case()
    reduced, rep = active_reduce(circ, obs_strings)
    assert reduced.n_qubits == 2  # compaction really renumbered

    report_path = tmp_path / "report.json"
    qasm_path = tmp_path / "reduced.qasm"
    report_path.write_text(rep.to_json(), encoding="utf-8")
    qasm_path.write_text(dumps_qasm(reduced), encoding="utf-8")

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    r = subprocess.run(
        [
            sys.executable, "-c", _CONSUMER.format(src=src),
            str(report_path), str(qasm_path),
        ],
        capture_output=True, text=True, check=True,
    )
    consumer_value = float(r.stdout.strip())

    # ground truth: the ORIGINAL circuit with the ORIGINAL observable
    (original_value,) = expectations(circ, [Observable.parse("Z2*Z3")])
    assert abs(consumer_value - original_value) <= (
        rep.certified_bounds[0] + 1e-12
    )
    assert consumer_value == pytest.approx(original_value, abs=1e-12)
