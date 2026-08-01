# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Independent reduction verifier — a deliberately simple second opinion.

This file exists so ACTIVE's claims can be checked without trusting the
reducer: it re-computes expectation values of the original and reduced
circuits by the most naive method that exists and checks the reducer's
claim. It shares NO logic with the reduction engine:

* the reducer (reduce.py/passes.py/tracker.py) works *backward* with
  Pauli transfer algebra and never builds a state;
* the test simulator (sim.py) works *forward* with tensor contractions;
* this file builds every gate as an explicit full-register 2^n x 2^n
  matrix and multiplies it into the state vector — brute force, nothing
  to hide a correlated bug in.

Never on the production path: ``active_reduce`` touches it only under an
explicit ``verify=True``, which accepts exponential cost on purpose. Not
the CERTIFY primitive either: it checks ACTIVE's own claim on one
concrete reduction; it proves nothing about general circuit equivalence.

Complexity: O(4^n) time per gate, O(4^n) memory. Refuses circuits with
more than MAX_VERIFY_QUBITS (10) qubits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .caches import cache_put
from .errors import ActiveError
from .ir import Circuit
from .paulis import Distribution, Measurement, Observable
from .report import Report

MAX_VERIFY_QUBITS = 10

# Absolute slack added to every comparison, covering float noise of the
# brute-force matrix products themselves (empirically ~1e-13 at 10 qubits).
FLOAT_SLACK = 1e-9

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class VerificationError(ActiveError):
    """The independent check failed — do not trust the reduction."""


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one independent check (all values recomputed from scratch)."""

    ok: bool
    observable_names: list[str]
    original_expectations: list[float]
    reduced_expectations: list[float]
    differences: list[float]
    allowed: list[float]  # certified bound + FLOAT_SLACK per observable

    def summary(self) -> str:
        lines = ["independent verification: " + ("PASS" if self.ok else "FAIL")]
        for n, o, r, d, a in zip(
            self.observable_names,
            self.original_expectations,
            self.reduced_expectations,
            self.differences,
            self.allowed,
        ):
            mark = "ok" if d <= a else "VIOLATION"
            lines.append(
                f"  {n}: original {o:+.12f}  reduced {r:+.12f}  "
                f"|diff| {d:.3e} <= allowed {a:.3e}  [{mark}]"
            )
        return "\n".join(lines)


_PERM_CACHE: dict[tuple, np.ndarray] = {}


def _basis_permutation(order: tuple[int, ...], n: int) -> np.ndarray:
    """Index map of the basis permutation sending bit t of an index to
    qubit position order[t]: entry idx -> permuted index. This IS the
    explicit permutation matrix, stored as one index per basis state
    instead of a dense 0/1 matrix — the same object, stored compactly."""
    key = (order, n)
    hit = _PERM_CACHE.get(key)
    if hit is not None:
        return hit
    dim = 2**n
    perm = np.empty(dim, dtype=np.int64)
    for idx in range(dim):
        out = 0
        for t in range(n):
            bit = (idx >> (n - 1 - t)) & 1
            out |= bit << (n - 1 - order[t])
        perm[idx] = out
    cache_put(_PERM_CACHE, key, perm, 10_000)
    return perm


def _final_state(circuit: Circuit) -> np.ndarray:
    """The state the circuit produces from |0...0>, computed the dumb way.

    Every gate is expanded to a full-register matrix: Kronecker product
    with identity (gate qubits leading), then conjugated by an explicit
    basis permutation that moves each bit to its actual qubit position
    (applied as a gather/scatter of amplitudes — a permutation matrix in
    index form). Qubit 0 is the most significant bit, matching the package
    convention.
    """
    n = circuit.n_qubits
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    for gate in circuit.gates:
        u = np.asarray(gate.unitary(), dtype=complex)
        extra = [q for q in range(n) if q not in gate.qubits]
        order = tuple(gate.qubits) + tuple(extra)
        big = np.kron(u, np.eye(2 ** len(extra), dtype=complex))
        perm = _basis_permutation(order, n)
        # apply P big P^T: gather (P^T), multiply, scatter (P)
        t = state[perm]
        t = big @ t
        out = np.empty_like(state)
        out[perm] = t
        state = out
    return state


def _observable_matrix(obs: Observable, n_qubits: int) -> np.ndarray:
    m = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for coeff, term in obs.terms:
        p = np.array([[1.0 + 0j]])
        for q in range(n_qubits):
            p = np.kron(p, _PAULI[term.axis(q)])
        m = m + coeff * p
    return m


def _expectation(circuit: Circuit, obs: Observable) -> float:
    state = _final_state(circuit)
    o = _observable_matrix(obs, circuit.n_qubits)
    val = np.vdot(state, o @ state)
    if abs(val.imag) > 1e-9:
        raise VerificationError(
            "non-real expectation value; observable is not Hermitian"
        )
    return float(val.real)


def _distribution(circuit: Circuit, qubits: tuple[int, ...]) -> np.ndarray:
    """Joint outcome distribution of measuring the given qubits, the dumb
    way: sum |amplitude|^2 over every basis state, bucketed by the measured
    bits (first listed qubit = most significant outcome bit)."""
    state = _final_state(circuit)
    n = circuit.n_qubits
    k = len(qubits)
    probs = np.zeros(2**k)
    for idx in range(2**n):
        p = abs(state[idx]) ** 2
        out = 0
        for t, q in enumerate(qubits):
            bit = (idx >> (n - 1 - q)) & 1
            out |= bit << (k - 1 - t)
        probs[out] += p
    return probs


def verify_reduction(
    original: Circuit,
    original_observables: list[Observable],
    reduced: Circuit,
    report: Report,
    *,
    raise_on_failure: bool = True,
) -> VerificationResult:
    """Independently confirm a reduction against its report.

    Args:
        original: The circuit that was reduced.
        original_observables: The measurements as given for the original
            circuit (original qubit numbering): Observables and/or
            Distributions. For a Distribution the check is that EVERY
            outcome probability moved by at most the certified bound.
        reduced: The reduced circuit returned by ``active_reduce``.
        report: Its report; ``report.observables`` (reduced numbering) and
            ``report.certified_bounds`` are the claims under test.
        raise_on_failure: Raise :class:`VerificationError` on any violation
            (default). If False, return the failing result instead.

    Returns:
        A :class:`VerificationResult` with every recomputed number.

    Raises:
        VerificationError: The check failed, or the circuit is too large to
            brute-force (> MAX_VERIFY_QUBITS qubits).
    """
    if original.n_qubits > MAX_VERIFY_QUBITS:
        raise VerificationError(
            f"cannot brute-force verify {original.n_qubits} qubits "
            f"(limit {MAX_VERIFY_QUBITS}); verification is exponential "
            "by design"
        )
    if len(original_observables) != len(report.observables):
        raise VerificationError(
            "report observable count does not match the input observables"
        )
    names, orig_e, red_e, diffs, allowed = [], [], [], [], []
    ok = True
    for obs_in, obs_out, bound, name in zip(
        original_observables,
        report.observables,
        report.certified_bounds,
        report.observable_names,
    ):
        if isinstance(obs_in, Distribution):
            if not isinstance(obs_out, Distribution):
                raise VerificationError(
                    f"measurement {name}: report type mismatch"
                )
            p0 = _distribution(original, obs_in.qubits)
            p1 = _distribution(reduced, obs_out.qubits)
            # the claim under test: EVERY outcome probability within bound
            worst = int(np.argmax(np.abs(p0 - p1)))
            e0, e1 = float(p0[worst]), float(p1[worst])
            d = float(np.max(np.abs(p0 - p1)))
        else:
            e0 = _expectation(original, obs_in)
            e1 = _expectation(reduced, obs_out)
            d = abs(e0 - e1)
        a = bound + FLOAT_SLACK
        names.append(name)
        orig_e.append(e0)
        red_e.append(e1)
        diffs.append(d)
        allowed.append(a)
        if d > a:
            ok = False
    result = VerificationResult(
        ok=ok,
        observable_names=names,
        original_expectations=orig_e,
        reduced_expectations=red_e,
        differences=diffs,
        allowed=allowed,
    )
    if not ok and raise_on_failure:
        raise VerificationError(result.summary())
    return result
