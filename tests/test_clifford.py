"""Clifford resynthesis: exactness, keep-if-smaller, zero effect elsewhere."""

import numpy as np
import pytest

from conftest import expectations, random_circuit, random_observable
from qxc_active import Circuit, active_reduce
from qxc_active.clifford import (
    CLIFFORD_NAMES,
    clifford_prefix_length,
    simulate_prefix,
    synthesize_preparation,
)
from qxc_active.paulis import coerce_observables
from qxc_active.verifier import verify_reduction
from qxc_active import sim


class TestTableauSynthesis:
    @pytest.mark.parametrize("seed", range(25))
    def test_prep_reproduces_state_exactly(self, seed):
        """Synthesized preparation must produce the identical state (up to
        global phase) for random Clifford circuits."""
        rng = np.random.default_rng(50000 + seed)
        n = int(rng.integers(1, 9))
        circ = random_circuit(
            rng, n, int(rng.integers(0, 15 * n)), clifford_only=True
        )
        prep = synthesize_preparation(simulate_prefix(n, circ.gates))
        assert prep is not None
        pc = Circuit(n)
        for g in prep:
            pc.append(g)
        inner = np.vdot(sim.statevector(circ), sim.statevector(pc))
        assert abs(abs(inner) - 1.0) < 1e-9

    def test_prep_of_zero_state_is_empty_or_tiny(self):
        prep = synthesize_preparation(simulate_prefix(4, []))
        assert prep is not None
        pc = Circuit(4)
        for g in prep:
            pc.append(g)
        psi = sim.statevector(pc)
        assert abs(abs(psi[0]) - 1.0) < 1e-12

    def test_prefix_detection(self):
        c = (
            Circuit(3)
            .add("h", 0)
            .add("cx", (0, 1))
            .add("rz", 2, 0.3)
            .add("cz", (1, 2))
        )
        assert clifford_prefix_length(c.gates) == 2

    def test_matrix_gate_ends_prefix(self):
        c = Circuit(1)
        c.add("h", 0)
        c.add("custom", 0, matrix=np.array([[0, 1], [1, 0]], dtype=complex))
        assert clifford_prefix_length(c.gates) == 1


class TestResynthesisInPipeline:
    @pytest.mark.parametrize("seed", range(10))
    def test_dense_clifford_reduced_and_verified(self, seed):
        rng = np.random.default_rng(51000 + seed)
        n = int(rng.integers(3, 8))
        circ = random_circuit(rng, n, 20 * n, clifford_only=True)
        obs = coerce_observables(
            [f"Z{q}" for q in range(n)]  # the hard (global) regime
        )
        reduced, report = active_reduce(circ, obs)
        verify_reduction(circ, obs, reduced, report)
        # dense Clifford at 20n gates: state prep needs O(n^2); must win
        assert reduced.n_gates < circ.n_gates
        assert any(
            r.reason == "clifford_resynthesis" for r in report.removed_gates
        )
        # exact rewrite: certified bound stays at float-dust scale
        assert all(b < 1e-9 for b in report.certified_bounds)

    def test_flag_off_disables(self):
        rng = np.random.default_rng(99)
        n = 5
        circ = random_circuit(rng, n, 20 * n, clifford_only=True)
        obs = [f"Z{q}" for q in range(n)]
        red_on, rep_on = active_reduce(circ, obs)
        red_off, rep_off = active_reduce(circ, obs, clifford_resynthesis=False)
        assert not any(
            r.reason == "clifford_resynthesis" for r in rep_off.removed_gates
        )
        assert red_on.n_gates <= red_off.n_gates

    def test_non_clifford_circuit_unchanged_by_flag(self):
        """On rotation circuits the pass must change nothing at all."""
        rng = np.random.default_rng(7)
        circ = Circuit(4)
        for q in range(4):
            circ.add("ry", q, 0.3 + q)
        for q in range(3):
            circ.add("cx", (q, q + 1))
        obs = ["Z0*Z3"]
        red_on, rep_on = active_reduce(circ, obs)
        red_off, rep_off = active_reduce(circ, obs, clifford_resynthesis=False)
        assert red_on == red_off

    def test_keep_if_smaller_small_prefix_untouched(self):
        """GHZ-3 prep is already minimal: synthesis (which costs ~n^2 gates)
        must not replace it."""
        c = Circuit(3).add("h", 0).add("cx", (0, 1)).add("cx", (1, 2)).add(
            "rz", 2, 0.4
        )
        reduced, report = active_reduce(c, ["X0*X1*X2"])
        assert not any(
            r.reason == "clifford_resynthesis" for r in report.removed_gates
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_mixed_circuit_exactness_preserved(self, seed):
        """Clifford prefix + rotation suffix: full pipeline stays exact."""
        rng = np.random.default_rng(52000 + seed)
        n = int(rng.integers(3, 7))
        circ = random_circuit(rng, n, 15 * n, clifford_only=True)
        for q in range(n):
            circ.add("rz", q, float(rng.uniform(-2, 2)))
        for q in range(n - 1):
            circ.add("cx", (q, q + 1))
        obs_in = [random_observable(rng, n)]
        reduced, report = active_reduce(circ, obs_in)
        obs = coerce_observables(obs_in)
        before = expectations(circ, obs)
        after = expectations(reduced, report.observables)
        for b, a in zip(before, after):
            assert abs(b - a) <= 1e-8
        verify_reduction(circ, obs, reduced, report)

    def test_idempotent_with_resynthesis(self):
        rng = np.random.default_rng(3)
        n = 6
        circ = random_circuit(rng, n, 20 * n, clifford_only=True)
        obs = [f"Z{q}" for q in range(n)]
        reduced, report = active_reduce(circ, obs)
        again, report2 = active_reduce(reduced, report.observables)
        assert again == reduced

    def test_approx_mode_with_resynthesis_bound_holds(self):
        rng = np.random.default_rng(4)
        n = 5
        circ = random_circuit(rng, n, 15 * n, clifford_only=True)
        for q in range(n):
            circ.add("rx", q, 1e-5)
        obs_in = [random_observable(rng, n)]
        reduced, report = active_reduce(circ, obs_in, epsilon=0.01)
        verify_reduction(circ, coerce_observables(obs_in), reduced, report)

    def test_all_clifford_names_have_tableau_semantics(self):
        """Every name in CLIFFORD_NAMES must actually be a Clifford gate the
        tableau can process (guards future registry edits)."""
        for name in sorted(CLIFFORD_NAMES):
            from qxc_active.gates import REGISTRY

            spec = REGISTRY[name]
            c = Circuit(spec.n_qubits)
            c.add(name, tuple(range(spec.n_qubits)))
            simulate_prefix(spec.n_qubits, c.gates)  # must not raise
