"""Independent cross-check of certified bounds against Qiskit.

These tests validate ACTIVE's central claim WITHOUT going through
``qxc_active.sim`` or ``qxc_active.verifier``: both the original and the
reduced circuit are rebuilt in Qiskit (via ``tools/qiskit_bridge.py``, a
measuring instrument that is never installed or imported by shipped code)
and evaluated with Qiskit's ``Statevector`` and ``SparsePauliOp``. The
measured difference must lie within the certified bound.

Also checks that our OpenQASM 3 export means the same thing to an outside
parser (Qiskit's qasm3 importer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import UnitaryGate  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from qxc_active import (  # noqa: E402
    Circuit,
    Distribution,
    Gate,
    active_reduce,
    dumps_qasm,
    loads_qasm,
)
from qxc_active.paulis import Observable, coerce_observables  # noqa: E402

from conftest import random_circuit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qiskit_bridge import (  # noqa: E402
    from_qiskit,
    observable_from_qiskit,
    to_qiskit,
    to_sparse_pauli_op,
)

# Slack for Qiskit's own float noise on top of the certified bound (same
# scale the in-repo verifier uses).
SLACK = 1e-9


def qk_expectation(circuit: Circuit, obs: Observable) -> float:
    """<obs> on the circuit's output state, computed entirely by Qiskit."""
    sv = Statevector(to_qiskit(circuit))
    val = sv.expectation_value(to_sparse_pauli_op(obs, circuit.n_qubits))
    assert abs(val.imag) < 1e-9
    return float(val.real)


def assert_bound_holds(circuit: Circuit, observables, epsilon=0.0, **kw):
    """Reduce, then check every |<O>_orig - <O>_red| <= certified bound."""
    reduced, report = active_reduce(circuit, observables, epsilon=epsilon, **kw)
    originals = coerce_observables(observables)
    for obs_in, obs_out, bound, name in zip(
        originals, report.observables, report.certified_bounds,
        report.observable_names,
    ):
        e0 = qk_expectation(circuit, obs_in)
        e1 = qk_expectation(reduced, obs_out)
        assert abs(e0 - e1) <= bound + SLACK, (
            f"{name}: qiskit says |{e0} - {e1}| = {abs(e0 - e1)} "
            f"> certified {bound}"
        )
    return reduced, report


OBSERVABLE_SHAPES = [
    ["Z0"],
    ["X1*Z4"],
    ["IZXYZI"],  # dense, qubit 0 leftmost
    ["0.5*Z0*Z1 + 0.25*X2 - Y0"],
    ["Z0", "X0*X1", "0.5*Y2*Z3 - 1.5*Z5"],
    [" + ".join(f"Z{q}" for q in range(6))],  # global support
]


@pytest.mark.parametrize("obs", OBSERVABLE_SHAPES)
def test_exact_mode_random_universal(obs):
    rng = np.random.default_rng(20260729)
    for seed_extra in range(3):
        circ = random_circuit(
            np.random.default_rng(1000 + seed_extra), 6, 40,
            include_matrix_gates=True,
        )
        assert_bound_holds(circ, obs)
    del rng


def test_exact_mode_ghz():
    circ = Circuit(6).add("h", 0)
    for q in range(5):
        circ.add("cx", (q, q + 1))
    assert_bound_holds(circ, ["Z0", "Z0*Z5", "X0*X1*X2*X3*X4*X5"])


@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.3])
def test_approximate_mode_budget(epsilon):
    for seed in range(3):
        circ = random_circuit(np.random.default_rng(2000 + seed), 6, 50)
        _, report = assert_bound_holds(
            circ, ["Z0", "0.5*Z1*Z2 + 0.5*X3"], epsilon=epsilon
        )
        assert all(a <= epsilon + 1e-15 for a in report.approx_bounds)


def test_forced_coarsening_at_min_max_terms():
    # rotation-dense circuit + wide observable forces the term count past
    # the documented minimum cap of 16; the coarsened analysis must stay
    # sound, which is exactly what the Qiskit cross-check verifies
    circ = random_circuit(np.random.default_rng(3000), 7, 60)
    _, report = assert_bound_holds(
        circ,
        [" + ".join(f"Z{q}" for q in range(7)), "X0*Y3*Z6"],
        max_terms=16,
    )


def test_forced_truncation_in_approx_mode():
    circ = random_circuit(np.random.default_rng(3100), 7, 60)
    assert_bound_holds(
        circ, [" + ".join(f"Z{q}" for q in range(7))],
        epsilon=0.3, max_terms=16,
    )


def test_distribution_outcome_probabilities():
    # per-outcome probability bound, cross-checked with Qiskit's
    # Statevector.probabilities. Endianness: ACTIVE's Distribution puts the
    # first (lowest-index) measured qubit at the MOST significant outcome
    # bit; Qiskit's probabilities(qargs) puts qargs[0] at the LEAST
    # significant — hence the reversed qargs list.
    for seed in range(3):
        circ = random_circuit(np.random.default_rng(4000 + seed), 5, 30)
        reduced, report = active_reduce(
            circ, [Distribution([0, 2, 3])], epsilon=0.05
        )
        dist_out = report.observables[0]
        p0 = Statevector(to_qiskit(circ)).probabilities(
            qargs=list(reversed([0, 2, 3]))
        )
        p1 = Statevector(to_qiskit(reduced)).probabilities(
            qargs=list(reversed(list(dist_out.qubits)))
        )
        bound = report.certified_bounds[0]
        assert np.max(np.abs(p0 - p1)) <= bound + SLACK


def _bitrev_perm(k: int) -> np.ndarray:
    """Index permutation between ACTIVE (qubit 0 = MSB) and Qiskit
    (qubit 0 = LSB) matrix conventions — an involution."""
    return np.array(
        [int(f"{i:0{k}b}"[::-1], 2) for i in range(2**k)], dtype=np.int64
    )


def _to_qiskit_convention(m: np.ndarray) -> np.ndarray:
    k = int(m.shape[0]).bit_length() - 1
    p = _bitrev_perm(k)
    return m[np.ix_(p, p)]


# fixed, arbitrary, non-special test angles per parameter count
_TEST_PARAMS = (0.37, -1.23, 0.81, 0.56)


class TestGateDefinitionsAgainstQiskit:
    """Every registry gate's matrix equals Qiskit's own gate of the same
    name — exactly, including global phase (atol only covers float
    noise of the two independent constructions)."""

    def test_every_registry_gate_matches(self):
        from qiskit.circuit.library.standard_gates import (
            get_standard_gate_name_mapping,
        )
        from qiskit.quantum_info import Operator

        from qxc_active.gates import REGISTRY, build_unitary

        mapping = get_standard_gate_name_mapping()
        missing = sorted(set(REGISTRY) - set(mapping))
        assert not missing, f"no independent definition for: {missing}"
        for name, spec in sorted(REGISTRY.items()):
            params = _TEST_PARAMS[: spec.n_params]
            ours = _to_qiskit_convention(
                np.asarray(build_unitary(name, params))
            )
            proto = mapping[name]
            theirs = (
                Operator(type(proto)(*params)).data
                if params
                else Operator(proto).data
            )
            assert np.allclose(ours, theirs, atol=1e-13), (
                f"gate {name!r} differs from Qiskit's definition "
                "(including global phase)"
            )


def _active_circuit_unitary(circ) -> np.ndarray:
    from qxc_active.gates import embed

    n = circ.n_qubits
    u = np.eye(2**n, dtype=complex)
    for g in circ.gates:
        u = embed(g.unitary(), g.qubits, tuple(range(n))) @ u
    return u


class TestQelib1ExtraGatesAgainstQiskit:
    """The injected qelib1 composites (u0, rccx, rc3x, c3x, c3sqrtx, c4x)
    expand to the exact unitary of Qiskit's corresponding gate — including
    global phase."""

    @pytest.mark.parametrize(
        "name,n_qubits,qk_factory",
        [
            ("rccx", 3, "RCCXGate"),
            ("rc3x", 4, "RC3XGate"),
            ("c3x", 4, "C3XGate"),
            ("c3sqrtx", 4, "C3SXGate"),
            ("c4x", 5, "C4XGate"),
        ],
    )
    def test_composite_matches_qiskit_gate(self, name, n_qubits, qk_factory):
        import qiskit.circuit.library as lib
        from qiskit.quantum_info import Operator

        args = ",".join(f"q[{i}]" for i in range(n_qubits))
        src = (
            f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n_qubits}];\n'
            f"{name} {args};\n"
        )
        ours = _to_qiskit_convention(
            _active_circuit_unitary(loads_qasm(src))
        )
        theirs = Operator(getattr(lib, qk_factory)()).data
        assert np.allclose(ours, theirs, atol=1e-12), (
            f"qelib1 {name!r} expansion differs from Qiskit's "
            f"{qk_factory} (including global phase)"
        )

    def test_u0_is_exact_identity(self):
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
            "u0(2.31) q[0];\n"
        )
        ours = _active_circuit_unitary(loads_qasm(src))
        assert np.allclose(ours, np.eye(2), atol=1e-15)


class TestFromQiskit:
    """The outside-in direction: circuits born in Qiskit. Round-tripping
    our own output cannot catch a symmetric convention error (it cancels
    between export and import); these tests can."""

    def test_z_on_qiskit_qubit_0_lands_on_active_qubit_0(self):
        from qiskit.quantum_info import SparsePauliOp as SPO

        # Qiskit label "IZ": Z on qubit 0 (rightmost character)
        obs = observable_from_qiskit(SPO("IZ"))
        assert len(obs.terms) == 1
        coeff, term = obs.terms[0]
        assert coeff == 1.0
        assert term.support_qubits() == [0]
        assert term.axis(0) == "Z"
        # and a gate: x on Qiskit qubit 0 is x on ACTIVE qubit 0
        qc = QuantumCircuit(2)
        qc.x(0)
        c = from_qiskit(qc)
        assert [(g.name, g.qubits) for g in c.gates] == [("x", (0,))]

    def test_raw_matrix_cx_equals_named_cx(self):
        from qiskit.circuit.library import CXGate
        from qiskit.quantum_info import Operator

        cx_matrix = Operator(CXGate()).data  # explicit 4x4, Qiskit order
        for control, target in [(0, 1), (1, 0), (0, 2)]:
            qc_m = QuantumCircuit(3)
            qc_m.append(UnitaryGate(cx_matrix), [control, target])
            qc_n = QuantumCircuit(3)
            qc_n.cx(control, target)
            u_m = _active_circuit_unitary(from_qiskit(qc_m))
            u_n = _active_circuit_unitary(from_qiskit(qc_n))
            assert np.allclose(u_m, u_n, atol=1e-13), (
                f"raw-matrix CX({control},{target}) != named cx"
            )

    def test_roundtrip_preserves_gate_sequence(self):
        circ = Circuit(4)
        circ.add("h", 0).add("cx", (0, 1)).add("rz", 2, 0.375)
        circ.add("cp", (1, 3), -0.5).add("swap", (0, 3))
        circ.add("ccx", (0, 1, 2)).add("rxx", (2, 3), 1.25)
        circ.add("crz", (3, 0), 0.7)
        back = from_qiskit(to_qiskit(circ))
        assert [(g.name, g.qubits, g.params) for g in back.gates] == [
            (g.name, g.qubits, g.params) for g in circ.gates
        ]

    def test_roundtrip_matrix_gate(self):
        from conftest import random_unitary

        rng = np.random.default_rng(8100)
        mat = random_unitary(rng, 4)
        circ = Circuit(3)
        circ.append(Gate("custom", (2, 0), (), matrix=mat))
        back = from_qiskit(to_qiskit(circ))
        assert len(back.gates) == 1
        g = back.gates[0]
        assert g.qubits == (2, 0)
        assert np.allclose(g.matrix, mat, atol=1e-15)

    def test_outside_in_bound_verification(self):
        """A circuit born in Qiskit (incl. a Haar-random UnitaryGate),
        imported, reduced, and verified against Qiskit's evaluation of
        the ORIGINAL — the test a symmetric convention error cannot
        pass."""
        from conftest import random_unitary
        from qiskit.quantum_info import SparsePauliOp as SPO

        rng = np.random.default_rng(8200)
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.37, 2)
        qc.append(UnitaryGate(random_unitary(rng, 4)), [1, 3])
        qc.ccx(2, 3, 4)
        qc.cp(-0.81, 0, 4)
        qc.ry(1.1, 3)
        qc.swap(1, 2)
        qc.t(0)
        qc.rzz(0.55, 3, 4)
        spo = SPO(["ZIIII", "IIXZI", "IYIIZ"], [0.5, -1.25, 0.75])

        circ = from_qiskit(qc)
        obs = observable_from_qiskit(spo)
        for epsilon in (0.0, 0.05):
            reduced, report = active_reduce(circ, [obs], epsilon=epsilon)
            # original side: Qiskit evaluates the FOREIGN circuit directly
            e0 = Statevector(qc).expectation_value(spo).real
            e1 = (
                Statevector(to_qiskit(reduced))
                .expectation_value(
                    to_sparse_pauli_op(
                        report.observables[0], reduced.n_qubits
                    )
                )
                .real
            )
            bound = report.certified_bounds[0]
            assert abs(e0 - e1) <= bound + SLACK, (
                f"outside-in: |{e0} - {e1}| > certified {bound}"
            )

    def test_measure_and_barrier_dropped_by_default(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)
        c = from_qiskit(qc)
        assert [g.name for g in c.gates] == ["h"]

    def test_ignore_measure_false_raises(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with pytest.raises(ValueError, match="measurement"):
            from_qiskit(qc, ignore_measure=False)

    def test_reset_raises_clearly(self):
        qc = QuantumCircuit(1)
        qc.reset(0)
        with pytest.raises(ValueError, match="not a unitary operation"):
            from_qiskit(qc)

    def test_control_flow_raises_clearly(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.x(0)
        with pytest.raises(ValueError, match="classical control flow"):
            from_qiskit(qc)

    def test_unbound_parameter_raises_clearly(self):
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(1)
        qc.rx(Parameter("t"), 0)
        with pytest.raises(ValueError, match="Bind any free Parameters"):
            from_qiskit(qc)

    def test_complex_observable_coefficient_rejected(self):
        from qiskit.quantum_info import SparsePauliOp as SPO

        with pytest.raises(ValueError, match="Hermitian"):
            observable_from_qiskit(SPO(["XY"], [1.0 + 0.5j]))


def test_matrix_gate_endianness_pinned_to_named_gate():
    # a raw-matrix CX (ACTIVE convention: first operand = most significant
    # bit, control first) must translate to exactly Qiskit's named cx on
    # the same operands — this pins the operand-reversal rule to an
    # external ground truth so a systematic endianness error cannot cancel
    # between the original and reduced circuit
    from qxc_active.gates import build_unitary

    cx_matrix = np.asarray(build_unitary("cx", ()))
    for control, target in [(0, 1), (1, 0), (0, 2), (2, 0)]:
        with_matrix = Circuit(3).add("h", 0).add("h", 1).add("h", 2)
        with_matrix.append(
            Gate("custom", (control, target), (), matrix=cx_matrix)
        )
        named = Circuit(3).add("h", 0).add("h", 1).add("h", 2)
        named.add("cx", (control, target))
        assert Statevector(to_qiskit(with_matrix)).equiv(
            Statevector(to_qiskit(named))
        ), f"matrix-CX({control},{target}) != named cx"


def test_qasm3_export_means_the_same_to_qiskit():
    # our OpenQASM 3 export, parsed by an OUTSIDE parser, must produce the
    # same state (up to global phase) as the direct bridge translation
    from qiskit import qasm3

    for seed in range(3):
        circ = random_circuit(np.random.default_rng(5000 + seed), 5, 30)
        sv_bridge = Statevector(to_qiskit(circ))
        sv_qasm = Statevector(qasm3.loads(dumps_qasm(circ)))
        assert sv_bridge.equiv(sv_qasm), (
            "OpenQASM 3 export disagrees with the direct translation"
        )


def test_reduced_qasm3_export_bound_via_outside_parser():
    # full pipeline through the integration surface: reduce, export QASM 3,
    # re-import with Qiskit's parser, evaluate — bound must still hold
    from qiskit import qasm3

    circ = random_circuit(np.random.default_rng(6000), 5, 30)
    obs = ["Z0", "0.5*Z1*Z2 + 0.25*X3"]
    reduced, report = active_reduce(circ, obs, epsilon=0.05)
    originals = coerce_observables(obs)
    sv_orig = Statevector(qasm3.loads(dumps_qasm(circ)))
    sv_red = Statevector(qasm3.loads(dumps_qasm(reduced)))
    for obs_in, obs_out, bound in zip(
        originals, report.observables, report.certified_bounds
    ):
        e0 = sv_orig.expectation_value(
            to_sparse_pauli_op(obs_in, circ.n_qubits)
        ).real
        e1 = sv_red.expectation_value(
            to_sparse_pauli_op(obs_out, reduced.n_qubits)
        ).real
        assert abs(e0 - e1) <= bound + SLACK
