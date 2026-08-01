"""THE approximate-mode guarantee: the certified bound is a TRUE upper bound.

For many random circuits we brute-force the actual expectation-value error of
the reduced circuit and assert it never exceeds the claimed certified bound,
and that the approximate part of the bound never exceeds the user's epsilon.
"""

import numpy as np
import pytest

from conftest import expectations, random_circuit, random_observable
from qxc_active import active_reduce
from qxc_active.paulis import coerce_observables

# simulator float-noise allowance on the *measured* error (the bound itself
# is analytic); statevector sims of ~40 gates carry ~1e-13 noise.
SIM_NOISE = 1e-9

EPSILONS = [1e-4, 1e-3, 1e-2, 0.1, 0.5]


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_actual_error_never_exceeds_certified_bound(seed, epsilon):
    rng = np.random.default_rng(9000 + seed)
    n = int(rng.integers(2, 6))
    circ = random_circuit(rng, n, int(rng.integers(5, 40)))
    obs_in = [random_observable(rng, n) for _ in range(rng.integers(1, 3))]
    reduced, report = active_reduce(circ, obs_in, epsilon=epsilon)

    obs = coerce_observables(obs_in)
    before = expectations(circ, obs)
    after = expectations(reduced, report.observables)
    for name, b, a, bound, approx in zip(
        report.observable_names,
        before,
        after,
        report.certified_bounds,
        report.approx_bounds,
    ):
        actual = abs(b - a)
        assert actual <= bound + SIM_NOISE, (
            f"BOUND VIOLATION observable {name}: actual error {actual!r} "
            f"exceeds certified bound {bound!r} (epsilon={epsilon})"
        )
        assert approx <= epsilon + 1e-15, (
            f"BUDGET VIOLATION: approx bound {approx!r} > epsilon {epsilon!r}"
        )


@pytest.mark.parametrize("seed", range(10))
def test_small_rotations_get_pruned(seed):
    """Approx mode must actually do something: circuits salted with tiny
    rotations should lose at least those under a generous budget."""
    rng = np.random.default_rng(500 + seed)
    n = 4
    circ = random_circuit(rng, n, 15)
    for q in range(n):
        circ.add("rz", q, 1e-7 * float(rng.uniform(0.1, 1.0)))
        circ.add("rx", q, 1e-7 * float(rng.uniform(0.1, 1.0)))
    obs = [random_observable(rng, n)]
    _, exact_report = active_reduce(circ, obs)
    _, approx_report = active_reduce(circ, obs, epsilon=1e-3)
    assert (
        approx_report.output_metrics["gates"]
        <= exact_report.output_metrics["gates"]
    )
    # with an epsilon of 1e-3, rotations of angle ~1e-7 are always affordable
    assert approx_report.n_gates_removed > exact_report.n_gates_removed


@pytest.mark.parametrize("seed", range(6))
def test_truncation_under_budget_is_certified(seed):
    """Force term-cap truncation (approx mode) and verify the bound holds."""
    rng = np.random.default_rng(7100 + seed)
    n = 5
    circ = random_circuit(rng, n, 40)
    obs_in = [random_observable(rng, n)]
    reduced, report = active_reduce(circ, obs_in, epsilon=0.3, max_terms=16)
    obs = coerce_observables(obs_in)
    before = expectations(circ, obs)
    after = expectations(reduced, report.observables)
    for b, a, bound in zip(before, after, report.certified_bounds):
        assert abs(b - a) <= bound + SIM_NOISE
    assert all(c <= 0.3 + 1e-15 for c in report.approx_bounds)


def test_epsilon_zero_equals_exact_mode():
    rng = np.random.default_rng(42)
    circ = random_circuit(rng, 4, 25)
    obs = [random_observable(rng, 4)]
    r0 = active_reduce(circ, obs, epsilon=0.0)[1]
    assert r0.theta == 0.0 and all(c == 0.0 for c in r0.approx_bounds)


def test_invalid_epsilon_rejected():
    from qxc_active import Circuit, InvalidOptionError

    c = Circuit(1).add("h", 0)
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(InvalidOptionError):
            active_reduce(c, ["Z0"], epsilon=bad)
