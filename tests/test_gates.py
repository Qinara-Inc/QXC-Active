"""Gate library: unitarity, embedding, identity detection, symmetry claims."""

import numpy as np
import pytest

from qxc_active.gates import (
    MERGEABLE_ROTATIONS,
    REGISTRY,
    SYMMETRIC_GATES,
    build_unitary,
    deviation_from_phase_identity,
    embed,
    is_unitary,
)
from qxc_active.ir import Gate

_SAMPLE_PARAMS = [0.37, -1.2, 2.9, 0.0]


def _sample(spec):
    return tuple(_SAMPLE_PARAMS[: spec.n_params])


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registry_gate_is_unitary(name):
    spec = REGISTRY[name]
    u = build_unitary(name, _sample(spec))
    assert u.shape == (2**spec.n_qubits, 2**spec.n_qubits)
    assert is_unitary(u)


@pytest.mark.parametrize("name", sorted(SYMMETRIC_GATES))
def test_symmetric_gates_really_are_symmetric(name):
    """Gates we merge across swapped qubit order must be swap-invariant."""
    spec = REGISTRY[name]
    u = build_unitary(name, _sample(spec))
    qubits = tuple(range(spec.n_qubits))
    swapped = (qubits[1], qubits[0]) + qubits[2:]
    assert np.allclose(embed(u, qubits, qubits), embed(u, swapped, qubits))


@pytest.mark.parametrize("name", sorted(MERGEABLE_ROTATIONS))
def test_mergeable_rotations_really_merge(name):
    """U(a) @ U(b) must equal U(a+b) for every declared rotation family."""
    a, b = 0.53, -1.17
    u_ab = build_unitary(name, (a,)) @ build_unitary(name, (b,))
    u_sum = build_unitary(name, (a + b,))
    assert np.allclose(u_ab, u_sum, atol=1e-12)


class TestEmbed:
    def test_identity_extension(self):
        cx = build_unitary("cx", ())
        full = embed(cx, (0, 1), (0, 1, 2))
        assert np.allclose(full, np.kron(cx, np.eye(2)))

    def test_permutation(self):
        # cx acting on (2, 0) inside qubits (0, 1, 2): apply to basis states
        cx = build_unitary("cx", ())
        full = embed(cx, (2, 0), (0, 1, 2))
        # |q0 q1 q2> = |001>: control q2=1 -> flips q0 -> |101>
        vec = np.zeros(8)
        vec[0b001] = 1.0
        out = full @ vec
        assert abs(out[0b101] - 1.0) < 1e-12

    def test_embed_same_qubits_is_identity_op(self):
        u = build_unitary("rzz", (0.7,))
        assert np.allclose(embed(u, (0, 1), (0, 1)), u)


class TestPhaseIdentity:
    def test_exact_identity(self):
        dev, _ = deviation_from_phase_identity(np.eye(4))
        assert dev < 1e-15

    def test_global_phase_identity(self):
        dev, phase = deviation_from_phase_identity(np.eye(2) * np.exp(1.3j))
        assert dev < 1e-12 and abs(phase - np.exp(1.3j)) < 1e-12

    def test_rz_two_pi_is_phase_identity(self):
        dev, _ = deviation_from_phase_identity(build_unitary("rz", (2 * np.pi,)))
        assert dev < 1e-12

    def test_not_identity(self):
        dev, _ = deviation_from_phase_identity(build_unitary("h", ()))
        assert dev > 0.5


class TestGateValidation:
    def test_custom_matrix_gate(self):
        g = Gate("mygate", (0,), (), matrix=np.array([[0, 1], [1, 0]]))
        assert np.allclose(g.unitary(), [[0, 1], [1, 0]])

    def test_non_unitary_matrix_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("bad", (0,), (), matrix=np.array([[1, 1], [0, 1]]))

    def test_wrong_matrix_shape_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("bad", (0, 1), (), matrix=np.eye(2))

    def test_unknown_gate_without_matrix_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("nosuchgate", (0,))

    def test_wrong_arity_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("cx", (0,))

    def test_wrong_param_count_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("rz", (0,), (0.1, 0.2))

    def test_duplicate_qubits_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("cx", (1, 1))

    def test_nonfinite_param_rejected(self):
        from qxc_active import InvalidCircuitError

        with pytest.raises(InvalidCircuitError):
            Gate("rz", (0,), (float("nan"),))
