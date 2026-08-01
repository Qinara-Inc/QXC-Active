"""Shared test helpers: seeded random circuits, observables, verification."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import qxc_active
from qxc_active import Circuit, Gate, Observable, PauliTerm
from qxc_active import sim

# Fail loudly if the qxc_active being tested is NOT this checkout: an
# editable install elsewhere on sys.path wins silently and the suite then
# passes without testing your edits (this has caught out one reviewer of
# this codebase already). QXC_ALLOW_EXTERNAL_IMPORT=1 is the escape hatch
# for deliberately testing an installed copy.
_expected_pkg = Path(__file__).resolve().parents[1] / "src" / "qxc_active"
_actual_pkg = Path(qxc_active.__file__).resolve().parent
if (
    _actual_pkg != _expected_pkg
    and os.environ.get("QXC_ALLOW_EXTERNAL_IMPORT") != "1"
):
    raise RuntimeError(
        f"tests imported qxc_active from {_actual_pkg}, not this checkout "
        f"({_expected_pkg}). You are NOT testing the code you are editing. "
        "Fix the environment (pip install -e . into the venv running "
        "pytest), or set QXC_ALLOW_EXTERNAL_IMPORT=1 to override on "
        "purpose."
    )

# (name, n_qubits, n_params) pools used by the random generator
GATE_POOL_1Q = [
    ("h", 1, 0), ("x", 1, 0), ("y", 1, 0), ("z", 1, 0), ("s", 1, 0),
    ("sdg", 1, 0), ("t", 1, 0), ("tdg", 1, 0), ("sx", 1, 0),
    ("rx", 1, 1), ("ry", 1, 1), ("rz", 1, 1), ("p", 1, 1), ("u", 1, 3),
]
GATE_POOL_2Q = [
    ("cx", 2, 0), ("cz", 2, 0), ("cy", 2, 0), ("ch", 2, 0), ("swap", 2, 0),
    ("iswap", 2, 0), ("cp", 2, 1), ("crx", 2, 1), ("cry", 2, 1),
    ("crz", 2, 1), ("rxx", 2, 1), ("ryy", 2, 1), ("rzz", 2, 1),
]
GATE_POOL_3Q = [("ccx", 3, 0), ("ccz", 3, 0), ("cswap", 3, 0)]

CLIFFORD_POOL = [
    ("h", 1, 0), ("s", 1, 0), ("sdg", 1, 0), ("x", 1, 0), ("z", 1, 0),
    ("cx", 2, 0), ("cz", 2, 0), ("swap", 2, 0),
]


def random_unitary(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Haar-ish random unitary via QR decomposition."""
    m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(m)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_circuit(
    rng: np.random.Generator,
    n_qubits: int,
    n_gates: int,
    include_3q: bool = True,
    include_matrix_gates: bool = False,
    clifford_only: bool = False,
) -> Circuit:
    """A seeded random circuit over a broad universal gate set."""
    if clifford_only:
        pool = list(CLIFFORD_POOL)
    else:
        pool = GATE_POOL_1Q + GATE_POOL_2Q
        if include_3q and n_qubits >= 3:
            pool = pool + GATE_POOL_3Q
    circ = Circuit(n_qubits)
    for _ in range(n_gates):
        if include_matrix_gates and rng.random() < 0.08 and n_qubits >= 2:
            k = int(rng.integers(1, 3))
            qubits = tuple(rng.choice(n_qubits, size=k, replace=False).tolist())
            circ.append(
                Gate("custom", qubits, (), matrix=random_unitary(rng, 2**k))
            )
            continue
        name, arity, n_params = pool[rng.integers(0, len(pool))]
        if arity > n_qubits:
            name, arity, n_params = "h", 1, 0
        qubits = tuple(rng.choice(n_qubits, size=arity, replace=False).tolist())
        params = tuple(rng.uniform(-2 * np.pi, 2 * np.pi, size=n_params).tolist())
        circ.add(name, qubits, params)
    return circ


def random_observable(
    rng: np.random.Generator, n_qubits: int, max_terms: int = 3
) -> Observable:
    """A seeded random Hermitian observable (weighted Pauli sum)."""
    n_terms = int(rng.integers(1, max_terms + 1))
    terms = []
    for _ in range(n_terms):
        weight = int(rng.integers(1, min(3, n_qubits) + 1))
        qubits = rng.choice(n_qubits, size=weight, replace=False)
        axes = {int(q): "XYZ"[rng.integers(0, 3)] for q in qubits}
        coeff = float(rng.uniform(-2.0, 2.0))
        if abs(coeff) < 1e-3:
            coeff = 1.0
        terms.append((coeff, PauliTerm.from_axes(axes)))
    return Observable(terms)


def expectations(circuit: Circuit, observables) -> list[float]:
    """Brute-force expectation of each observable on the circuit output."""
    psi = sim.statevector(circuit)
    return [
        sim.expectation_of_state(psi, o, circuit.n_qubits) for o in observables
    ]
