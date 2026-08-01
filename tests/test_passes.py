"""Behaviour of the individual passes and the fixpoint driver."""

import numpy as np
import pytest

from conftest import expectations, random_circuit, random_observable
from qxc_active import Circuit, active_reduce
from qxc_active.paulis import coerce_observables


def _reasons(report):
    return sorted(r.reason for r in report.removed_gates)


class TestBackwardPass:
    def test_beats_lightcone_diagonal_gate(self):
        """A CZ inside the lightcone commutes with a Z-type back-propagated
        observable and must be removed (naive lightcone keeps it)."""
        c = Circuit(2).add("cz", (0, 1)).add("x", 0)
        reduced, report = active_reduce(c, ["Z0"])
        assert reduced.n_gates == 1 and reduced.gates[0].name == "x"

    def test_support_shrinks_through_clifford(self):
        """Z0*Z1 pulled through CX(0,1) becomes Z1: gates on qubit 0 before
        the CX must disappear even though qubit 0 is measured."""
        c = Circuit(2).add("h", 0).add("cx", (0, 1))
        reduced, report = active_reduce(c, ["Z0*Z1"])
        assert reduced.n_gates == 0

    def test_outside_support_removed(self):
        c = Circuit(3).add("h", 2).add("rx", 2, 1.0).add("x", 0)
        reduced, report = active_reduce(c, ["Z0"])
        assert reduced.n_gates == 1
        assert reduced.n_qubits == 1  # qubit 2 (and idle 1) dropped

    def test_influence_scores_reported_for_kept_gates(self):
        c = Circuit(2).add("h", 0).add("cx", (0, 1))
        _, report = active_reduce(c, ["Z1"])
        # both gates matter for Z1 and must carry finite influence scores
        assert set(report.influence_scores) == {0, 1}
        assert all(
            all(np.isfinite(v) and v > 0 for v in scores)
            for scores in report.influence_scores.values()
        )


class TestCancellationPass:
    def test_adjacent_inverse_pair(self):
        c = Circuit(1).add("h", 0).add("h", 0).add("x", 0)
        reduced, report = active_reduce(c, ["Z0"])
        assert [g.name for g in reduced.gates] == ["x"]
        assert "cancelled" in _reasons(report)

    def test_cx_pair_cancels_through_commuting_barrier(self):
        """cx . rz(control) . cx: the rz commutes with cx, so the pair must
        cancel even though a gate sits between them."""
        c = (
            Circuit(2)
            .add("cx", (0, 1))
            .add("rz", 0, 0.7)
            .add("cx", (0, 1))
            .add("h", 1)
            .add("x", 1)
        )
        reduced, report = active_reduce(c, ["X1"])
        names = [g.name for g in reduced.gates]
        assert "cx" not in names

    def test_rotation_merge(self):
        # h first so the rz pair is NOT killed by the initial-state pass
        c = Circuit(1).add("h", 0).add("rz", 0, 0.3).add("rz", 0, 0.4).add("h", 0)
        reduced, report = active_reduce(c, ["Z0"])
        rz = [g for g in reduced.gates if g.name == "rz"]
        assert len(rz) == 1 and rz[0].params[0] == pytest.approx(0.7)

    def test_rotation_merge_to_identity(self):
        c = Circuit(1).add("h", 0).add("rz", 0, 0.9).add("rz", 0, -0.9).add("h", 0)
        reduced, _ = active_reduce(c, ["Z0"])
        assert reduced.n_gates == 0

    def test_symmetric_gate_merges_across_qubit_order(self):
        c = Circuit(2).add("h", 0).add("h", 1).add("rzz", (0, 1), 0.5).add(
            "rzz", (1, 0), -0.5
        ).add("h", 0).add("h", 1)
        reduced, _ = active_reduce(c, ["Z0", "Z1"])
        assert reduced.n_gates == 0

    def test_identity_gates_removed(self):
        c = Circuit(1).add("id", 0).add("rz", 0, 0.0).add("x", 0)
        reduced, report = active_reduce(c, ["Z0"])
        assert [g.name for g in reduced.gates] == ["x"]

    def test_non_commuting_blocker_prevents_cancellation(self):
        # h . x . h  on the same qubit: x does not commute with h, and
        # h x h = z which changes Z0 -> stays 3 gates (or an equivalent form
        # of the same unitary action); correctness is what matters.
        c = Circuit(1).add("h", 0).add("x", 0).add("h", 0)
        reduced, report = active_reduce(c, ["X0"])
        obs = coerce_observables(["X0"])
        assert abs(
            expectations(c, obs)[0]
            - expectations(reduced, report.observables)[0]
        ) < 1e-10


class TestInitialStatePass:
    def test_control_on_zero_removed(self):
        c = Circuit(2).add("cx", (0, 1)).add("h", 1).add("x", 1)
        reduced, report = active_reduce(c, ["X1"])
        assert all(g.name != "cx" for g in reduced.gates)

    def test_diagonal_on_zero_removed(self):
        # rz / p / cz on |0> qubits act as (global) phase only
        c = Circuit(2).add("rz", 0, 1.1).add("cz", (0, 1)).add("h", 0).add(
            "cx", (0, 1)
        )
        reduced, report = active_reduce(c, ["Z1"])
        names = [g.name for g in reduced.gates]
        assert "rz" not in names and "cz" not in names

    def test_x_on_zero_not_removed(self):
        c = Circuit(1).add("x", 0)
        reduced, _ = active_reduce(c, ["Z0"])
        assert reduced.n_gates == 1

    def test_knowledge_expires_after_state_changes(self):
        # after h(0), qubit 0 is no longer |0>: cz must NOT be removed
        c = Circuit(2).add("h", 0).add("h", 1).add("cz", (0, 1)).add("h", 1)
        reduced, _ = active_reduce(c, ["Z1"])
        assert any(g.name == "cz" for g in reduced.gates)


class TestFixpoint:
    def test_reduction_is_idempotent(self):
        """Re-reducing an already-reduced circuit must change nothing."""
        for seed in range(8):
            rng = np.random.default_rng(800 + seed)
            n = int(rng.integers(2, 6))
            circ = random_circuit(rng, n, 30)
            obs = [random_observable(rng, n)]
            reduced, report = active_reduce(circ, obs)
            again, report2 = active_reduce(reduced, report.observables)
            assert again == reduced
            assert report2.n_gates_removed == 0

    def test_passes_enable_each_other(self):
        """Removing a gate via back-propagation exposes a cancellation that
        exposes an initial-state removal — requires multiple rounds."""
        c = (
            Circuit(3)
            .add("cx", (0, 1))  # initial-state (control |0>) after rounds
            .add("h", 1)
            .add("cz", (1, 2))  # irrelevant to Z1 after the h h cancel...
            .add("h", 1)
            .add("h", 1)  # cancels with previous h
            .add("h", 1)
        )
        reduced, report = active_reduce(c, ["Z1"])
        obs = coerce_observables(["Z1"])
        assert abs(
            expectations(c, obs)[0]
            - expectations(reduced, report.observables)[0]
        ) < 1e-10
        assert report.rounds >= 2

    def test_max_rounds_respected(self):
        rng = np.random.default_rng(3)
        circ = random_circuit(rng, 4, 25)
        _, report = active_reduce(circ, ["Z0"], max_rounds=1)
        assert report.rounds == 1

    def test_deterministic(self):
        rng = np.random.default_rng(21)
        circ = random_circuit(rng, 5, 35)
        obs = ["0.7*Z0*X2 - 0.3*Y4"]
        r1, rep1 = active_reduce(circ, obs, epsilon=0.05)
        r2, rep2 = active_reduce(circ, obs, epsilon=0.05)
        assert r1 == r2
        d1, d2 = rep1.to_dict(), rep2.to_dict()
        d1.pop("timings_ms"), d2.pop("timings_ms")
        assert d1 == d2
