"""Conversion helpers between qxc_active and Qiskit — a measuring
instrument for the cross-check test suite, not an integration.

This module lives OUTSIDE ``src/``: it is never installed, never imported
by shipped code, and must never be. The package's user-facing import path
is OpenQASM 2 and 3 (native); this bridge exists only so an independent
stack (Qiskit's ``Statevector`` / ``SparsePauliOp``) can re-evaluate
ACTIVE's claims without touching ``qxc_active.sim`` or
``qxc_active.verifier``.

Both directions exist for a reason: round-tripping our own output
(``from_qiskit(to_qiskit(c))``) is a WEAKER test than importing a foreign
circuit, because a systematic convention error (e.g. reversed qubit
significance) applied symmetrically on export and import cancels out and
hides. Only an outside-in test — a circuit born in Qiskit, imported,
reduced, and verified against Qiskit's own evaluation of the original —
catches that class of bug. It has already produced a batch of false
soundness violations during a review of this codebase.

Endianness (the one thing easy to get wrong): qubit *indices* mean the
same thing in both packages, so gate arguments carry over unchanged
(``cx`` takes control first in both). The packages differ only in which
end of a state-vector index a qubit occupies — ACTIVE puts qubit 0 at the
most significant bit, Qiskit at the least. That matters in exactly two
places:

* raw-matrix gates: the operand list is reversed when appending the
  matrix to the Qiskit circuit;
* Pauli labels: Qiskit's leftmost label character is the *highest* qubit.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp

from qxc_active import Circuit, Gate
from qxc_active.gates import REGISTRY
from qxc_active.paulis import Observable, PauliTerm

# ACTIVE gate name -> Qiskit QuantumCircuit method of the same semantics.
# Qubit arguments are positionally identical in every one of these.
_DIRECT = {
    "id": "id", "x": "x", "y": "y", "z": "z", "h": "h", "s": "s",
    "sdg": "sdg", "t": "t", "tdg": "tdg", "sx": "sx", "sxdg": "sxdg",
    "rx": "rx", "ry": "ry", "rz": "rz", "p": "p", "u": "u",
    "cx": "cx", "cy": "cy", "cz": "cz", "ch": "ch", "csx": "csx",
    "cp": "cp", "crx": "crx", "cry": "cry", "crz": "crz",
    "rxx": "rxx", "ryy": "ryy", "rzz": "rzz", "swap": "swap",
    "iswap": "iswap", "ccx": "ccx", "ccz": "ccz", "cswap": "cswap",
}


def to_qiskit(circuit) -> QuantumCircuit:
    """Rebuild a qxc_active Circuit as a Qiskit QuantumCircuit."""
    qc = QuantumCircuit(circuit.n_qubits)
    for gate in circuit.gates:
        if gate.matrix is not None:
            # ACTIVE: first operand = most significant bit of the matrix
            # index; Qiskit: first operand = least significant. Reverse.
            qc.append(UnitaryGate(gate.matrix), list(reversed(gate.qubits)))
            continue
        name, params, qubits = gate.name, gate.params, list(gate.qubits)
        if name in _DIRECT:
            getattr(qc, _DIRECT[name])(*params, *qubits)
        elif name == "u1":
            qc.p(params[0], *qubits)
        elif name == "u2":
            qc.u(math.pi / 2, params[0], params[1], *qubits)
        elif name == "u3":
            qc.u(*params, *qubits)
        elif name == "cu1":
            qc.cp(params[0], *qubits)
        elif name == "cu3":
            qc.cu(params[0], params[1], params[2], 0.0, *qubits)
        elif name == "cu":
            qc.cu(*params, *qubits)
        else:
            # fall back to the gate's own unitary, operands reversed
            qc.append(
                UnitaryGate(gate.unitary()), list(reversed(gate.qubits))
            )
    return qc


# qiskit op names ACTIVE cannot treat as unitary gates
_CONTROL_FLOW = {"if_else", "while_loop", "for_loop", "switch_case"}

# qiskit legacy names whose ACTIVE registry spelling differs
_FROM_QISKIT_ALIASES = {"phase": "p", "cphase": "cp", "i": "id"}


def from_qiskit(qc: QuantumCircuit, *, ignore_measure: bool = True):
    """Import a Qiskit QuantumCircuit as a qxc_active Circuit.

    Named gates carry over by name and unchanged qubit indices (``cx``
    takes control first in both packages). Anything without a registry
    spelling — UnitaryGate, unknown or differently-parameterised gates —
    falls back to the gate's own matrix via ``op.to_matrix()``, with the
    operand list REVERSED (ACTIVE: first operand = most significant bit;
    Qiskit: least). ``op.params`` is not always a list of floats
    (UnitaryGate carries its matrix there; unbound Parameters appear
    too), so float conversion failures fall through to the matrix path.

    Measurements and barriers are dropped by default
    (``ignore_measure=False`` raises on measure instead). Classical
    control flow, classically-fed instructions and reset raise: ACTIVE
    reduces unitary circuits only — same scope as the QASM importers.
    """
    circ = Circuit(qc.num_qubits)
    for inst in qc.data:
        op = inst.operation
        name = _FROM_QISKIT_ALIASES.get(op.name, op.name)
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        if name == "barrier":
            continue
        if name == "measure":
            if ignore_measure:
                continue
            raise ValueError(
                "This circuit contains a measurement (ignore_measure="
                "False). ACTIVE reduces unitary circuits; measurements "
                "are specified separately as observables."
            )
        if name in _CONTROL_FLOW or getattr(op, "condition", None) is not None:
            raise ValueError(
                f"This circuit uses classical control flow ({op.name!r} "
                f"on qubits {qubits}).\n\nACTIVE reduces unitary "
                "circuits: its certified bounds are stated for a fixed "
                "sequence of gates, and a classically-controlled gate "
                "does not have one. Remove the conditional, or reduce "
                "the unitary part of the circuit separately."
            )
        if name == "reset":
            raise ValueError(
                f"This circuit uses reset (qubits {qubits}).\n\nACTIVE "
                "reduces unitary circuits: reset discards a qubit's "
                "state, which is not a unitary operation, so no "
                "certified bound covers it."
            )
        if inst.clbits:
            raise ValueError(
                f"Instruction {op.name!r} involves classical bits; "
                "ACTIVE reduces unitary circuits only."
            )
        params: tuple | None
        try:
            params = tuple(float(p) for p in op.params)
        except (TypeError, ValueError):
            params = None  # non-scalar params (matrix, unbound Parameter)
        spec = REGISTRY.get(name)
        if (
            params is not None
            and spec is not None
            and spec.n_qubits == len(qubits)
            and spec.n_params == len(params)
        ):
            circ.append(Gate(name, qubits, params))
            continue
        # matrix fallback: ACTIVE's first operand is the most significant
        # bit, Qiskit's the least — reverse the operand list
        try:
            matrix = np.asarray(op.to_matrix(), dtype=complex)
        except Exception as exc:
            raise ValueError(
                f"cannot import instruction {op.name!r}: it has no "
                f"float parameters and no concrete matrix ({exc}). "
                "Bind any free Parameters before importing."
            ) from exc
        circ.append(
            Gate(op.name, tuple(reversed(qubits)), (), matrix=matrix)
        )
    return circ


def observable_from_qiskit(op) -> Observable:
    """Import a Qiskit SparsePauliOp (or Pauli) as a qxc_active Observable.

    Qiskit's leftmost label character is the HIGHEST qubit, so labels are
    walked reversed. Complex coefficients are rejected: ACTIVE
    observables are Hermitian real-weighted Pauli sums.
    """
    spo = op if isinstance(op, SparsePauliOp) else SparsePauliOp(op)
    terms = []
    for label, coeff in spo.to_list():
        c = complex(coeff)
        if abs(c.imag) > 1e-12:
            raise ValueError(
                f"term {label!r} has complex coefficient {c}: ACTIVE "
                "observables are Hermitian real-weighted Pauli sums"
            )
        axes = {q: ch for q, ch in enumerate(reversed(label)) if ch != "I"}
        terms.append((c.real, PauliTerm.from_axes(axes)))
    return Observable(terms)


def to_sparse_pauli_op(observable, n_qubits: int) -> SparsePauliOp:
    """A qxc_active Observable as a Qiskit SparsePauliOp.

    Qiskit's leftmost label character is the highest qubit, so the label
    walks ACTIVE's qubits from n_qubits-1 down to 0.
    """
    labels = []
    coeffs = []
    for coeff, term in observable.terms:
        labels.append(
            "".join(term.axis(q) for q in range(n_qubits - 1, -1, -1))
        )
        coeffs.append(coeff)
    if not labels:  # numerically-zero observable
        return SparsePauliOp(["I" * n_qubits], [0.0])
    return SparsePauliOp(labels, coeffs)
