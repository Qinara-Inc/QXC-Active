"""THE exact-mode guarantee: reduction preserves every expectation value.

For random circuits small enough to simulate classically, the reduced
circuit's expectation values must equal the original's within numerical
tolerance (1e-8; the certified dust bounds are ~1e-12, the slack covers
simulator float noise).
"""

import numpy as np
import pytest

from conftest import expectations, random_circuit, random_observable
from qxc_active import active_reduce

TOL = 1e-8


def _check_exact(circ, observables):
    reduced, report = active_reduce(circ, observables)
    from qxc_active.paulis import coerce_observables

    obs = coerce_observables(observables)
    before = expectations(circ, obs)
    after = expectations(reduced, report.observables)
    for name, b, a, bound in zip(
        report.observable_names, before, after, report.certified_bounds
    ):
        assert abs(b - a) <= max(bound, 0.0) + TOL, (
            f"observable {name}: original {b!r} vs reduced {a!r} "
            f"(certified bound {bound!r})"
        )
    assert report.epsilon == 0.0
    assert all(c == 0.0 for c in report.approx_bounds)
    return reduced, report


@pytest.mark.parametrize("seed", range(40))
def test_random_universal_circuits(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(2, 7))
    circ = random_circuit(rng, n, int(rng.integers(5, 45)))
    observables = [random_observable(rng, n) for _ in range(rng.integers(1, 3))]
    _check_exact(circ, observables)


@pytest.mark.parametrize("seed", range(15))
def test_random_clifford_circuits(seed):
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(2, 8))
    circ = random_circuit(rng, n, int(rng.integers(10, 60)), clifford_only=True)
    observables = [random_observable(rng, n) for _ in range(rng.integers(1, 4))]
    _check_exact(circ, observables)


@pytest.mark.parametrize("seed", range(10))
def test_random_circuits_with_custom_matrix_gates(seed):
    """Universality: arbitrary user-supplied unitaries must be handled."""
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(2, 6))
    circ = random_circuit(
        rng, n, int(rng.integers(8, 30)), include_matrix_gates=True
    )
    observables = [random_observable(rng, n)]
    _check_exact(circ, observables)


@pytest.mark.parametrize("seed", range(8))
def test_low_term_cap_degrades_soundly(seed):
    """Even with an absurdly low term cap (forced coarsening) the result
    must stay exactly correct — degradation may only reduce aggressiveness."""
    rng = np.random.default_rng(4000 + seed)
    n = int(rng.integers(3, 6))
    circ = random_circuit(rng, n, 30)
    obs = [random_observable(rng, n)]
    reduced, report = active_reduce(circ, obs, max_terms=16)
    from qxc_active.paulis import coerce_observables

    before = expectations(circ, coerce_observables(obs))
    after = expectations(reduced, report.observables)
    for b, a in zip(before, after):
        assert abs(b - a) <= TOL


def test_string_and_dense_observables_accepted():
    rng = np.random.default_rng(5)
    circ = random_circuit(rng, 3, 12)
    _check_exact(circ, ["Z0", "IZX", "0.5*Z0*Z1 + 0.5*X2"])


@pytest.mark.parametrize("seed", range(6))
def test_compact_false_preserves_indices(seed):
    rng = np.random.default_rng(6000 + seed)
    n = 5
    circ = random_circuit(rng, n, 20)
    obs = [random_observable(rng, n)]
    reduced, report = active_reduce(circ, obs, compact=False)
    assert reduced.n_qubits == n
    from qxc_active.paulis import coerce_observables

    before = expectations(circ, coerce_observables(obs))
    after = expectations(reduced, report.observables)
    for b, a in zip(before, after):
        assert abs(b - a) <= TOL


def test_larger_clifford_circuit_stays_fast_and_exact():
    rng = np.random.default_rng(99)
    circ = random_circuit(rng, 10, 300, clifford_only=True)
    obs = ["Z0", "Z4*Z7"]
    reduced, report = active_reduce(circ, obs)
    from qxc_active.paulis import coerce_observables

    before = expectations(circ, coerce_observables(obs))
    after = expectations(reduced, report.observables)
    for b, a in zip(before, after):
        assert abs(b - a) <= TOL
