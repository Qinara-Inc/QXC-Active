"""Pauli transfer: the conjugation expansion must reproduce U^dag P U exactly."""

import numpy as np
import pytest

from conftest import random_unitary
from qxc_active.gates import build_unitary
from qxc_active.transfer import (
    conjugate_pauli,
    local_pauli_matrix,
    transfer_deviation,
)


def _reconstruct(transfer, k):
    terms, _dust = transfer
    m = np.zeros((2**k, 2**k), dtype=complex)
    for codes, d in terms:
        m += d * local_pauli_matrix(codes)
    return m


@pytest.mark.parametrize("name,params", [
    ("h", ()), ("s", ()), ("t", ()), ("sx", ()),
    ("rx", (0.7,)), ("rz", (-2.1,)), ("u", (0.3, 1.1, -0.6)),
    ("cx", ()), ("cz", ()), ("iswap", ()), ("rzz", (0.9,)),
    ("rxx", (1.3,)), ("cp", (0.4,)), ("ccx", ()), ("cswap", ()),
])
def test_transfer_matches_dense_conjugation(name, params):
    u = build_unitary(name, params)
    k = int(np.log2(u.shape[0]))
    for idx in range(1, 4**k):  # every non-identity local Pauli
        codes = []
        v = idx
        for _ in range(k):
            codes.append(v % 4)
            v //= 4
        codes = tuple(reversed(codes))
        tr = conjugate_pauli(u, codes, cache_key=(name, params))
        expected = u.conj().T @ local_pauli_matrix(codes) @ u
        assert np.allclose(_reconstruct(tr, k), expected, atol=1e-10)


def test_transfer_matches_dense_for_random_unitaries():
    rng = np.random.default_rng(7)
    for k in (1, 2, 3):
        for _ in range(5):
            u = random_unitary(rng, 2**k)
            for idx in (1, 4**k - 1, 4**k // 2 or 1):
                codes = []
                v = idx
                for _ in range(k):
                    codes.append(v % 4)
                    v //= 4
                codes = tuple(reversed(codes))
                tr = conjugate_pauli(u, codes)
                expected = u.conj().T @ local_pauli_matrix(codes) @ u
                assert np.allclose(_reconstruct(tr, k), expected, atol=1e-9)


def test_clifford_gates_give_single_term():
    for name in ("h", "s", "cx", "cz", "swap"):
        u = build_unitary(name, ())
        k = int(np.log2(u.shape[0]))
        for idx in range(1, 4**k):
            codes = []
            v = idx
            for _ in range(k):
                codes.append(v % 4)
                v //= 4
            codes = tuple(reversed(codes))
            terms, dust = conjugate_pauli(u, codes, cache_key=(name,))
            assert len(terms) == 1
            assert abs(abs(terms[0][1]) - 1.0) < 1e-12


def test_identity_input_passthrough():
    u = build_unitary("cx", ())
    terms, dust = conjugate_pauli(u, (0, 0), cache_key=("cx", ()))
    assert terms == (((0, 0), 1.0),) and dust == 0.0


def test_deviation_is_true_norm_bound():
    """transfer_deviation must upper-bound the exact ||U^dag P U - P||."""
    rng = np.random.default_rng(11)
    for k in (1, 2):
        for _ in range(20):
            u = random_unitary(rng, 2**k)
            idx = int(rng.integers(1, 4**k))
            codes = []
            v = idx
            for _ in range(k):
                codes.append(v % 4)
                v //= 4
            codes = tuple(reversed(codes))
            tr = conjugate_pauli(u, codes)
            p = local_pauli_matrix(codes)
            true_norm = np.linalg.norm(u.conj().T @ p @ u - p, 2)
            assert transfer_deviation(codes, tr) >= true_norm - 1e-10


def test_commuting_gate_has_zero_deviation():
    u = build_unitary("cz", ())
    codes = (3, 3)  # Z (x) Z commutes with CZ
    tr = conjugate_pauli(u, codes, cache_key=("cz", ()))
    assert transfer_deviation(codes, tr) < 1e-12
