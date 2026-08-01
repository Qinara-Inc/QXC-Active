"""Property-based tests (Hypothesis): the guarantees hold on arbitrary input.

Strategies generate random circuits over a universal gate set and random
weighted-sum observables; every property is one of the shipped guarantees:

* exact mode preserves all expectation values;
* approximate mode's certified bound is a true upper bound and its
  approximate part never exceeds epsilon;
* reduction is deterministic;
* reduction is idempotent (fixpoint reached).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from conftest import expectations
from qxc_active import Circuit, Observable, PauliTerm, active_reduce
from qxc_active.paulis import coerce_observables

MAX_QUBITS = 5

_GATE_CHOICES = [
    ("h", 1, 0), ("x", 1, 0), ("s", 1, 0), ("sdg", 1, 0), ("t", 1, 0),
    ("sx", 1, 0), ("rx", 1, 1), ("ry", 1, 1), ("rz", 1, 1), ("p", 1, 1),
    ("u", 1, 3), ("cx", 2, 0), ("cz", 2, 0), ("swap", 2, 0), ("iswap", 2, 0),
    ("cp", 2, 1), ("rzz", 2, 1), ("rxx", 2, 1), ("crx", 2, 1),
    ("ccx", 3, 0), ("ccz", 3, 0),
]

_angles = st.floats(
    min_value=-6.3, max_value=6.3, allow_nan=False, allow_infinity=False
)


@st.composite
def circuits(draw) -> Circuit:
    n = draw(st.integers(min_value=1, max_value=MAX_QUBITS))
    n_gates = draw(st.integers(min_value=0, max_value=25))
    circ = Circuit(n)
    for _ in range(n_gates):
        name, arity, n_params = draw(st.sampled_from(_GATE_CHOICES))
        if arity > n:
            name, arity, n_params = "h", 1, 0
        qubits = tuple(
            draw(
                st.lists(
                    st.integers(0, n - 1),
                    min_size=arity,
                    max_size=arity,
                    unique=True,
                )
            )
        )
        params = tuple(draw(_angles) for _ in range(n_params))
        circ.add(name, qubits, params)
    return circ


@st.composite
def observables_for(draw, n_qubits: int) -> Observable:
    n_terms = draw(st.integers(1, 2))
    terms = []
    for _ in range(n_terms):
        weight = draw(st.integers(1, min(3, n_qubits)))
        qubits = draw(
            st.lists(
                st.integers(0, n_qubits - 1),
                min_size=weight,
                max_size=weight,
                unique=True,
            )
        )
        axes = {q: draw(st.sampled_from("XYZ")) for q in qubits}
        coeff = draw(
            st.floats(
                min_value=0.05, max_value=2.0, allow_nan=False
            )
        ) * draw(st.sampled_from([1.0, -1.0]))
        terms.append((coeff, PauliTerm.from_axes(axes)))
    return Observable(terms)


@st.composite
def circuit_and_observables(draw):
    circ = draw(circuits())
    n_obs = draw(st.integers(1, 2))
    obs = [draw(observables_for(circ.n_qubits)) for _ in range(n_obs)]
    return circ, obs


@given(circuit_and_observables())
@settings(max_examples=60, deadline=None)
def test_exact_mode_preserves_expectations(case):
    circ, obs_in = case
    reduced, report = active_reduce(circ, obs_in)
    before = expectations(circ, coerce_observables(obs_in))
    after = expectations(reduced, report.observables)
    for b, a, bound in zip(before, after, report.certified_bounds):
        assert abs(b - a) <= bound + 1e-8
    assert all(c == 0.0 for c in report.approx_bounds)


@given(circuit_and_observables(), st.sampled_from([1e-3, 1e-2, 0.1, 0.5]))
@settings(max_examples=60, deadline=None)
def test_certified_bound_is_true_upper_bound(case, epsilon):
    circ, obs_in = case
    reduced, report = active_reduce(circ, obs_in, epsilon=epsilon)
    before = expectations(circ, coerce_observables(obs_in))
    after = expectations(reduced, report.observables)
    for b, a, bound, approx in zip(
        before, after, report.certified_bounds, report.approx_bounds
    ):
        assert abs(b - a) <= bound + 1e-9
        assert approx <= epsilon + 1e-15


@given(circuit_and_observables())
@settings(max_examples=25, deadline=None)
def test_reduction_is_deterministic(case):
    circ, obs_in = case
    r1, rep1 = active_reduce(circ, obs_in, epsilon=0.01)
    r2, rep2 = active_reduce(circ, obs_in, epsilon=0.01)
    assert r1 == r2
    d1, d2 = rep1.to_dict(), rep2.to_dict()
    d1.pop("timings_ms"), d2.pop("timings_ms")
    assert d1 == d2


@given(circuit_and_observables())
@settings(max_examples=25, deadline=None)
def test_reduction_is_idempotent(case):
    circ, obs_in = case
    reduced, report = active_reduce(circ, obs_in)
    again, report2 = active_reduce(reduced, report.observables)
    assert again == reduced
    assert report2.n_gates_removed == 0


@given(circuit_and_observables())
@settings(max_examples=25, deadline=None)
def test_metrics_never_increase(case):
    circ, obs_in = case
    _, report = active_reduce(circ, obs_in)
    for key in ("qubits", "gates", "depth", "two_qubit_gates"):
        assert report.output_metrics[key] <= report.input_metrics[key]
