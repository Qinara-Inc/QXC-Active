"""Distribution measurements: full sampling distribution preserved (exact)
or per-outcome-probability bounded (approx), plus many-observables stress."""

import numpy as np
import pytest

from conftest import random_circuit, random_observable
from qxc_active import (
    Circuit,
    Distribution,
    InvalidObservableError,
    active_reduce,
)
from qxc_active import sim
from qxc_active.paulis import coerce_observables
from qxc_active.verifier import verify_reduction


def marginal(circuit: Circuit, qubits: tuple[int, ...]) -> np.ndarray:
    """Reference joint distribution via the (independent) test simulator."""
    psi = sim.statevector(circuit)
    n = circuit.n_qubits
    k = len(qubits)
    out = np.zeros(2**k)
    for idx in range(2**n):
        p = abs(psi[idx]) ** 2
        o = 0
        for t, q in enumerate(qubits):
            o |= ((idx >> (n - 1 - q)) & 1) << (k - 1 - t)
        out[o] += p
    return out


class TestExactDistribution:
    @pytest.mark.parametrize("seed", range(20))
    def test_distribution_preserved_exactly(self, seed):
        rng = np.random.default_rng(60000 + seed)
        n = int(rng.integers(2, 6))
        circ = random_circuit(rng, n, int(rng.integers(5, 35)))
        k = int(rng.integers(1, n + 1))
        qubits = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        dist = Distribution(qubits)
        reduced, report = active_reduce(circ, [dist])
        out = report.observables[0]
        p0 = marginal(circ, qubits)
        p1 = marginal(reduced, out.qubits)
        assert np.max(np.abs(p0 - p1)) <= report.certified_bounds[0] + 1e-8
        # and the independent verifier agrees
        verify_reduction(circ, [dist], reduced, report)

    def test_ghz_full_distribution_keeps_everything(self):
        """Sampling all GHZ qubits: the whole preparation matters."""
        c = Circuit(4).add("h", 0)
        for q in range(3):
            c.add("cx", (q, q + 1))
        reduced, report = active_reduce(c, [Distribution(range(4))])
        p0 = marginal(c, (0, 1, 2, 3))
        p1 = marginal(reduced, report.observables[0].qubits)
        assert np.allclose(p0, p1, atol=1e-10)
        assert reduced.n_gates >= 4  # no free lunch on GHZ sampling

    def test_ghz_partial_distribution_drops_tail(self):
        """Sampling only qubit 0 of a GHZ chain: the tail CXs are provably
        irrelevant to that marginal."""
        c = Circuit(4).add("h", 0)
        for q in range(3):
            c.add("cx", (q, q + 1))
        reduced, report = active_reduce(c, [Distribution([0])])
        p0 = marginal(c, (0,))
        p1 = marginal(reduced, report.observables[0].qubits)
        assert np.allclose(p0, p1, atol=1e-10)
        assert reduced.n_gates == 1  # just the h

    def test_mixed_distribution_and_pauli(self):
        rng = np.random.default_rng(3)
        n = 5
        circ = random_circuit(rng, n, 25)
        obs = [Distribution([0, 2]), "Z1", "0.5*X3*X4"]
        reduced, report = active_reduce(circ, obs)
        verify_reduction(circ, coerce_observables(obs), reduced, report)
        assert len(report.certified_bounds) == 3

    def test_distribution_qubits_all_kept_active(self):
        c = Circuit(5).add("h", 0)
        reduced, report = active_reduce(c, [Distribution([0, 3, 4])])
        # sampled qubits survive qubit elimination even with no gates
        assert reduced.n_qubits == 3


class TestApproxDistribution:
    @pytest.mark.parametrize("seed", range(15))
    @pytest.mark.parametrize("epsilon", [1e-3, 1e-2, 0.2])
    def test_per_outcome_bound_holds(self, seed, epsilon):
        rng = np.random.default_rng(61000 + seed)
        n = int(rng.integers(2, 6))
        circ = random_circuit(rng, n, int(rng.integers(5, 30)))
        k = int(rng.integers(1, n + 1))
        qubits = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        dist = Distribution(qubits)
        reduced, report = active_reduce(circ, [dist], epsilon=epsilon)
        assert report.approx_bounds[0] <= epsilon + 1e-15
        p0 = marginal(circ, qubits)
        p1 = marginal(reduced, report.observables[0].qubits)
        actual = float(np.max(np.abs(p0 - p1)))
        assert actual <= report.certified_bounds[0] + 1e-9
        verify_reduction(circ, [dist], reduced, report)

    def test_group_budget_is_shared(self):
        """The distribution's generators share one epsilon budget: the
        summed generator charges (halved) must stay under epsilon, so the
        reported bound cannot exceed epsilon even with many qubits."""
        rng = np.random.default_rng(9)
        n = 5
        circ = random_circuit(rng, n, 30)
        for q in range(n):
            circ.add("ry", q, 1e-4)
        _, report = active_reduce(circ, [Distribution(range(n))], epsilon=0.01)
        assert report.approx_bounds[0] <= 0.01 + 1e-15


class TestManyObservables:
    def test_forty_observables_at_once(self):
        rng = np.random.default_rng(77)
        n = 6
        circ = random_circuit(rng, n, 40)
        obs = [random_observable(rng, n) for _ in range(40)]
        reduced, report = active_reduce(circ, obs)
        assert len(report.certified_bounds) == 40
        verify_reduction(circ, coerce_observables(obs), reduced, report)

    def test_hamiltonian_style_many_term_observable(self):
        rng = np.random.default_rng(78)
        n = 6
        circ = random_circuit(rng, n, 30)
        terms = " + ".join(
            f"{rng.uniform(0.1, 1):.3f}*Z{q}*Z{(q + 1) % n}" for q in range(n)
        )
        reduced, report = active_reduce(circ, [terms])
        verify_reduction(
            circ, coerce_observables([terms]), reduced, report
        )


class TestValidation:
    def test_bad_distributions_rejected(self):
        with pytest.raises(InvalidObservableError):
            Distribution([])
        with pytest.raises(InvalidObservableError):
            Distribution([0, 0])
        with pytest.raises(InvalidObservableError):
            Distribution([-1])

    def test_distribution_out_of_range_rejected(self):
        with pytest.raises(InvalidObservableError):
            active_reduce(Circuit(2).add("h", 0), [Distribution([5])])

    def test_cli_sample_qubits(self, tmp_path, capsys):
        from qxc_active.cli import main

        f = tmp_path / "c.qasm"
        f.write_text(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q;\n'
            "h q[0]; cx q[0], q[1]; cx q[1], q[2];\n",
            encoding="utf-8",
        )
        assert main(["reduce", str(f), "--sample-qubits", "0,1", "--json"]) == 0
        import json

        blob = json.loads(capsys.readouterr().out)
        assert blob["observables"] == ["dist(0,1)"]

    def test_cli_bad_sample_qubits(self, tmp_path, capsys):
        from qxc_active.cli import main

        f = tmp_path / "c.qasm"
        f.write_text(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; h q[0];',
            encoding="utf-8",
        )
        assert main(["reduce", str(f), "--sample-qubits", "a,b"]) == 2
