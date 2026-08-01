"""The hybrid dict/numpy tracker: promotion, equivalence, determinism.

The vectorised path only engages past PROMOTE_THRESHOLD terms; these tests
force promotion with a tiny threshold so every code path runs on small,
brute-force-verifiable circuits.
"""

from __future__ import annotations

import numpy as np
import pytest

import qxc_active.tracker as tracker_mod
from qxc_active import Circuit, active_reduce
from conftest import random_circuit


@pytest.fixture()
def force_promotion(monkeypatch):
    monkeypatch.setattr(tracker_mod, "PROMOTE_THRESHOLD", 4)


def _wide_obs(n: int) -> list[str]:
    return [" + ".join(f"Z{q}" for q in range(n)), "X0*Y1", "Z1*Z2"]


def _removals(report):
    return sorted(
        (r.index, r.name, tuple(r.qubits), r.reason)
        for r in report.removed_gates
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_promoted_run_matches_dict_run_decisions(seed, monkeypatch):
    circ = random_circuit(np.random.default_rng(7000 + seed), 6, 60)
    obs = _wide_obs(6)
    monkeypatch.setattr(tracker_mod, "PROMOTE_THRESHOLD", 10**9)  # dict only
    red_d, rep_d = active_reduce(circ, obs)
    monkeypatch.setattr(tracker_mod, "PROMOTE_THRESHOLD", 4)  # vec asap
    red_v, rep_v = active_reduce(circ, obs)
    # acceptance criterion: identical removal decisions, identical circuit
    assert _removals(rep_d) == _removals(rep_v)
    assert red_d.gates == red_v.gates
    # bounds legitimately differ across the promotion boundary: each run's
    # dust tallies the float crumbs THAT run discarded, and intermediate
    # values differ at ulp, so the crumb sets differ slightly (documented
    # in CHANGELOG.md). They must stay the same order of magnitude; actual
    # soundness of each bound is proven brute-force in the verify=True
    # tests below.
    for bd, bv in zip(rep_d.certified_bounds, rep_v.certified_bounds):
        assert bv <= 2 * bd + 1e-15
        assert bd <= 2 * bv + 1e-15


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_promoted_bounds_hold_by_brute_force(seed, force_promotion):
    circ = random_circuit(np.random.default_rng(7100 + seed), 6, 50)
    obs = _wide_obs(6)
    # verify=True runs the independent brute-force verifier and raises on
    # any bound violation — the soundness check for the vectorised path
    active_reduce(circ, obs, verify=True)


@pytest.mark.parametrize("epsilon", [0.05, 0.3])
def test_promoted_truncation_path(epsilon, force_promotion):
    # small max_terms + tiny promotion threshold: the tracker is
    # vectorised when truncation fires
    circ = random_circuit(np.random.default_rng(7200), 7, 60)
    reduced, rep = active_reduce(
        circ,
        [" + ".join(f"Z{q}" for q in range(7))],
        epsilon=epsilon,
        max_terms=16,
    )
    assert all(a <= epsilon + 1e-15 for a in rep.approx_bounds)
    # brute-force soundness check on the same configuration
    active_reduce(
        circ,
        [" + ".join(f"Z{q}" for q in range(7))],
        epsilon=epsilon,
        max_terms=16,
        verify=True,
    )


def test_promoted_coarsening_path(force_promotion):
    circ = random_circuit(np.random.default_rng(7300), 7, 60)
    _, rep = active_reduce(
        circ, [" + ".join(f"Z{q}" for q in range(7))], max_terms=16
    )
    # exact mode + cap: must coarsen (soundly), never truncate
    assert all(a == 0.0 for a in rep.approx_bounds)
    active_reduce(
        circ,
        [" + ".join(f"Z{q}" for q in range(7))],
        max_terms=16,
        verify=True,
    )


def test_promoted_run_is_deterministic(force_promotion):
    circ = random_circuit(np.random.default_rng(7400), 6, 60)
    obs = _wide_obs(6)
    red1, rep1 = active_reduce(circ, obs)
    red2, rep2 = active_reduce(circ, obs)
    assert red1.gates == red2.gates
    assert rep1.certified_bounds == rep2.certified_bounds
    assert _removals(rep1) == _removals(rep2)


def test_no_promotion_above_qubit_62():
    # qubit indices > 62 cannot be int64 masks: the dict path must be used
    # and still be correct. 70 qubits, gates concentrated high.
    circ = Circuit(70)
    for q in range(60, 70):
        circ.add("h", q)
    for q in range(60, 69):
        circ.add("cx", (q, q + 1))
    for q in range(60, 70):
        circ.add("t", q)
    _, rep = active_reduce(circ, ["Z69", "X60*X69"])
    assert rep.certified_bounds[0] < 1e-9  # exact mode, dust only
