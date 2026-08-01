# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Dense statevector simulator (verification scaffolding).

Used by the test suite to check ACTIVE's guarantees by brute force on
small circuits. It is intentionally simple and
independent of the reduction code: a second implementation path that the
reducer cannot "agree with by construction".
"""

from __future__ import annotations

import numpy as np

from .errors import InvalidCircuitError
from .ir import Circuit
from .paulis import Observable, PauliTerm
from .gates import PAULI_MATRICES

# 2^24 amplitudes (~256 MB complex128) is a hard safety ceiling.
MAX_SIM_QUBITS = 24


def statevector(circuit: Circuit) -> np.ndarray:
    """Final state of the circuit applied to |0...0>.

    Returns a flat complex vector of length ``2^n``; index bit convention:
    qubit 0 is the most significant bit of the index.
    """
    n = circuit.n_qubits
    if n > MAX_SIM_QUBITS:
        raise InvalidCircuitError(
            f"refusing to simulate {n} qubits (limit {MAX_SIM_QUBITS})"
        )
    state = np.zeros(2**n, dtype=complex) if n else np.ones(1, dtype=complex)
    if n:
        state[0] = 1.0
    psi = state.reshape((2,) * n) if n else state
    for gate in circuit.gates:
        psi = _apply(psi, gate.unitary(), gate.qubits, n)
    return psi.reshape(-1)


def _apply(
    psi: np.ndarray, u: np.ndarray, qubits: tuple[int, ...], n: int
) -> np.ndarray:
    k = len(qubits)
    ut = u.reshape((2,) * (2 * k))
    # contract input axes of the gate with the state's qubit axes
    psi = np.tensordot(ut, psi, axes=(list(range(k, 2 * k)), list(qubits)))
    # result axes: gate-out axes first, then remaining state axes; restore order
    return np.moveaxis(psi, list(range(k)), list(qubits))


def apply_pauli(psi: np.ndarray, term: PauliTerm, n: int) -> np.ndarray:
    """Apply a Pauli string to a state given as a flat vector."""
    t = psi.reshape((2,) * n) if n else psi
    for q in term.support_qubits():
        code = {"X": 1, "Y": 2, "Z": 3}[term.axis(q)]
        t = _apply(t, PAULI_MATRICES[code], (q,), n)
    return t.reshape(-1)


def expectation(circuit: Circuit, observable: Observable) -> float:
    """<0...0| U^dag O U |0...0> for the circuit's unitary U."""
    psi = statevector(circuit)
    return expectation_of_state(psi, observable, circuit.n_qubits)


def expectation_of_state(psi: np.ndarray, observable: Observable, n: int) -> float:
    """Expectation of a Hermitian Pauli-sum observable in a given state."""
    if observable.max_qubit() >= n:
        raise InvalidCircuitError(
            f"observable acts on qubit {observable.max_qubit()} but the "
            f"state has only {n} qubit(s)"
        )
    total = 0.0
    for coeff, term in observable.terms:
        val = np.vdot(psi, apply_pauli(psi, term, n))
        total += coeff * val.real
    return float(total)
