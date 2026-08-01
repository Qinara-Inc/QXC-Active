"""Edge cases: empty, single-qubit, all-inactive, fully-entangled, errors."""

import json

import numpy as np
import pytest

from conftest import expectations, random_circuit
from qxc_active import (
    Circuit,
    InvalidCircuitError,
    InvalidObservableError,
    InvalidOptionError,
    Observable,
    active_reduce,
)
from qxc_active.paulis import coerce_observables


def test_empty_circuit():
    reduced, report = active_reduce(Circuit(3), ["Z0"])
    assert reduced.n_gates == 0 and reduced.n_qubits == 1
    assert report.certified_bounds == [0.0]


def test_single_qubit_circuit():
    c = Circuit(1).add("h", 0).add("t", 0).add("h", 0)
    reduced, report = active_reduce(c, ["Z0"])
    obs = coerce_observables(["Z0"])
    assert abs(
        expectations(c, obs)[0] - expectations(reduced, report.observables)[0]
    ) < 1e-10


def test_all_gates_inactive():
    """Nothing touches the measured qubit: everything must vanish."""
    c = Circuit(4)
    for _ in range(3):
        c.add("h", 1).add("cx", (1, 2)).add("rz", 3, 0.4)
    reduced, report = active_reduce(c, ["Z0"])
    assert reduced.n_gates == 0
    assert reduced.n_qubits == 1
    assert report.output_metrics["qubits"] == 1


def test_fully_entangled_global_observable_no_reduction():
    """Adversarial case: every gate matters. ACTIVE must keep them all and
    still be exactly correct — honesty over aggression."""
    rng = np.random.default_rng(1234)
    n = 4
    c = Circuit(n)
    for layer in range(3):
        for q in range(n):
            c.add("ry", q, float(rng.uniform(0.4, 2.0)))
        for q in range(n - 1):
            c.add("cx", (q, q + 1))
    obs_str = "*".join(f"X{q}" for q in range(n))
    reduced, report = active_reduce(c, [obs_str])
    obs = coerce_observables([obs_str])
    assert abs(
        expectations(c, obs)[0] - expectations(reduced, report.observables)[0]
    ) < 1e-8
    # the brickwork genuinely feeds the global observable; nothing outside
    # support should exist to remove
    assert reduced.n_qubits == n


def test_identity_observable():
    """The identity Pauli has empty support: whole circuit is irrelevant."""
    c = Circuit(2).add("h", 0).add("cx", (0, 1))
    reduced, report = active_reduce(c, [Observable.parse("I")])
    assert reduced.n_gates == 0


def test_zero_qubit_circuit_with_identity_observable():
    reduced, report = active_reduce(Circuit(0), [Observable.parse("I")])
    assert reduced.n_qubits == 0 and reduced.n_gates == 0


def test_observable_out_of_range_rejected():
    with pytest.raises(InvalidObservableError):
        active_reduce(Circuit(2).add("h", 0), ["Z5"])


def test_bad_observable_strings_rejected():
    for bad in ["", "Q3", "Z0*Z0", "0.5"]:
        with pytest.raises(InvalidObservableError):
            active_reduce(Circuit(1).add("h", 0), [bad])


def test_no_observables_rejected():
    with pytest.raises(InvalidObservableError):
        active_reduce(Circuit(1).add("h", 0), [])


def test_bad_options_rejected():
    c = Circuit(1).add("h", 0)
    with pytest.raises(InvalidOptionError):
        active_reduce(c, ["Z0"], max_terms=2)
    with pytest.raises(InvalidOptionError):
        active_reduce(c, ["Z0"], cancellation_lookback=0)
    with pytest.raises(InvalidOptionError):
        active_reduce(c, ["Z0"], max_rounds=0)
    with pytest.raises(InvalidOptionError):
        active_reduce(c, ["Z0"], dust_tolerance=-1.0)
    with pytest.raises(InvalidOptionError):
        active_reduce("not a circuit", ["Z0"])  # type: ignore[arg-type]


def test_gate_index_out_of_circuit_rejected():
    c = Circuit(2)
    with pytest.raises(InvalidCircuitError):
        c.add("h", 5)


def test_report_is_json_serializable_and_complete():
    rng = np.random.default_rng(77)
    circ = random_circuit(rng, 4, 20)
    reduced, report = active_reduce(circ, ["Z0", "X1*X2"], epsilon=0.01)
    blob = json.loads(report.to_json())
    for key in (
        "version", "epsilon", "theta", "observables", "approx_bounds",
        "numerical_dust", "certified_bounds", "input", "output",
        "removed_gates", "removed_qubits", "qubit_map", "influence_scores",
        "coarsening_events", "max_terms_seen", "rounds", "timings_ms",
    ):
        assert key in blob, key
    assert blob["input"]["gates"] == 20
    assert report.summary()  # renders without error


def test_qubit_map_consistency():
    c = Circuit(6).add("h", 4).add("cx", (4, 5)).add("x", 0)
    reduced, report = active_reduce(c, ["Z4"])
    # only qubit 4's cone survives: h(4); qubit 5 & 0 gates die
    assert set(report.qubit_map.keys()) == {4}
    assert report.qubit_map[4] == 0
    assert report.observables[0].terms[0][1].support_qubits() == [0]


def test_duplicate_observables_ok():
    c = Circuit(2).add("h", 0).add("cx", (0, 1))
    reduced, report = active_reduce(c, ["Z1", "Z1"])
    assert len(report.certified_bounds) == 2
