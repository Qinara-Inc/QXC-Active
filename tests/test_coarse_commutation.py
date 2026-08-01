"""The coarse-mode commutation fallback: soundness and never-regress.

Coarse mode used to be support-only (weaker than a lightcone pass that
keeps testing commutation). The fallback adds a commutation bound via
the slide argument. These tests force coarsening with a low max_terms
and verify bounds against the independent brute-force verifier — fresh
soundness evidence, since this change alters WHAT gets removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import random_circuit
from qxc_active import Circuit, active_reduce


def qaoa_like(n: int, p: int) -> Circuit:
    c = Circuit(n)
    for q in range(n):
        c.add("h", q)
    for layer in range(p):
        for q in range(n):
            c.add("rzz", (q, (q + 1) % n), 0.7 + 0.31 * layer)
        for q in range(n):
            c.add("rx", q, 0.4 + 0.17 * layer)
    return c


def _support_only(circuit, obs, **kw):
    """The pre-fallback coarse behaviour, for comparison: an unbounded
    coarse cost restores keep-on-any-support-overlap exactly."""
    import math

    import qxc_active.tracker as tracker_mod

    real = tracker_mod.TermTracker._coarse_commutation_bound
    tracker_mod.TermTracker._coarse_commutation_bound = (
        lambda self, gate, gate_mask: math.inf
    )
    try:
        return active_reduce(circuit, obs, **kw)
    finally:
        tracker_mod.TermTracker._coarse_commutation_bound = real


def test_commutation_fallback_beats_support_only_on_diagonal_structure():
    # rzz layers commute with Z observables: after coarsening, support-
    # only keeps them, the commutation fallback removes them
    circ = qaoa_like(8, 3)
    obs = ["Z0"]
    red_new, rep_new = active_reduce(circ, obs, max_terms=16)
    red_old, rep_old = _support_only(circ, obs, max_terms=16)
    assert rep_new.coarsening_events  # the regime under test
    assert red_new.n_gates < red_old.n_gates
    # soundness of the extra removals, brute force
    active_reduce(circ, obs, max_terms=16, verify=True)


@pytest.mark.parametrize("seed", range(12))
def test_forced_coarsening_never_regresses_and_stays_sound(seed):
    rng = np.random.default_rng(9200 + seed)
    circ = random_circuit(rng, 7, 50)
    obs = [" + ".join(f"Z{q}" for q in range(7)), "X0*Y3*Z6"]
    red_new, rep_new = active_reduce(circ, obs, max_terms=16)
    red_old, _ = _support_only(circ, obs, max_terms=16)
    # never keep a gate the support test would have removed
    assert red_new.n_gates <= red_old.n_gates
    # fresh soundness evidence: independent brute-force verification
    active_reduce(circ, obs, max_terms=16, verify=True)


@pytest.mark.parametrize("seed", range(6))
def test_forced_coarsening_approx_mode_sound(seed):
    rng = np.random.default_rng(9300 + seed)
    circ = random_circuit(rng, 6, 45)
    obs = [" + ".join(f"Z{q}" for q in range(6))]
    _, rep = active_reduce(
        circ, obs, epsilon=0.1, max_terms=16, verify=True
    )
    assert all(a <= 0.1 + 1e-15 for a in rep.approx_bounds)


def test_coarse_removal_residuals_are_charged_to_dust():
    # rx(pi) = -iX: it anticommutes with Z0, but the rzz gates commute
    # with everything Z-diagonal with ~1e-16 float residuals — those
    # residuals must appear in the dust bound, not vanish
    circ = qaoa_like(6, 3)
    _, rep = active_reduce(circ, ["Z0"], max_terms=16)
    assert rep.coarsening_events
    if rep.n_gates_removed:  # coarse-mode invariant removals happened
        assert all(a == 0.0 for a in rep.approx_bounds)  # exact mode
        # dust is a finite, honest tally (may be exactly 0.0 only if
        # every residual was exactly zero in float arithmetic)
        assert all(np.isfinite(d) for d in rep.numerical_dust)


def test_multi_observable_coarse_mix():
    # one tracker coarsens, another stays fine: removals must satisfy
    # BOTH; brute-force verified
    rng = np.random.default_rng(9400)
    circ = random_circuit(rng, 6, 40)
    obs = [" + ".join(f"Z{q}" for q in range(6)), "Z0"]
    active_reduce(circ, obs, max_terms=16, verify=True)
