# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Exact Pauli transfer through a gate (Heisenberg conjugation).

For a gate ``G`` on ``k`` qubits and a Hermitian Pauli ``P`` restricted to
those qubits, this module computes the exact expansion

    G^dagger P G = sum_Q d_Q Q     (Q ranging over the 4^k local Paulis)

with real coefficients ``d_Q`` (real because conjugation preserves
Hermiticity and the Hermitian Pauli basis has real coordinates for Hermitian
operators). Results are cached per ``(gate, local Pauli)`` pair, which makes
repeated structure (the common case in real circuits) effectively free.

Local Paulis are encoded as a tuple of axis codes, one per gate qubit in the
gate's qubit order: 0=I, 1=X, 2=Y, 3=Z.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .caches import cache_put
from .gates import PAULI_MATRICES

LocalPauli = tuple[int, ...]
# (expansion terms, dropped-dust bound): dust is the sum of |coefficients|
# discarded as numerical noise; a provable bound on the norm of what was cut.
Transfer = tuple[tuple[tuple[LocalPauli, float], ...], float]

# Numerical dust below this is dropped from transfer expansions; every drop is
# accounted into the certified numerical-dust bound by callers.
TRANSFER_DUST = 1e-14

_BASIS_CACHE: dict[int, np.ndarray] = {}
_CODES_CACHE: dict[int, list[LocalPauli]] = {}
_TRANSFER_CACHE: dict[tuple, Transfer] = {}
_TRANSFER_CACHE_LIMIT = 500_000


def _pauli_basis(k: int) -> tuple[np.ndarray, list[LocalPauli]]:
    """Stack of all 4^k Hermitian Pauli matrices on k qubits + their codes."""
    if k in _BASIS_CACHE:
        return _BASIS_CACHE[k], _CODES_CACHE[k]
    mats = []
    codes: list[LocalPauli] = []
    for idx in range(4**k):
        # decode base-4 digits, first qubit = most significant digit
        digits = []
        v = idx
        for _ in range(k):
            digits.append(v % 4)
            v //= 4
        digits.reverse()
        m = np.array([[1.0 + 0j]])
        for d in digits:
            m = np.kron(m, PAULI_MATRICES[d])
        mats.append(m)
        codes.append(tuple(digits))
    stack = np.stack(mats)
    _BASIS_CACHE[k] = stack
    _CODES_CACHE[k] = codes
    return stack, codes


def local_pauli_matrix(codes: LocalPauli) -> np.ndarray:
    """Dense matrix of a local Pauli given axis codes (first = MSB)."""
    m = np.array([[1.0 + 0j]])
    for d in codes:
        m = np.kron(m, PAULI_MATRICES[d])
    return m


def conjugate_pauli(
    unitary: np.ndarray, codes: LocalPauli, cache_key: tuple | None = None
) -> Transfer:
    """Expansion of ``U^dagger P U`` in the local Hermitian Pauli basis.

    Args:
        unitary: The gate unitary (2^k x 2^k).
        codes: Axis codes of the input local Pauli, gate-qubit order.
        cache_key: Hashable identity of the unitary for memoisation.

    Returns:
        ``(terms, dust)``: terms is a tuple of ``(output_codes, coefficient)``
        sorted by descending magnitude; ``dust`` bounds the norm of the
        coefficients dropped as numerical noise (|coeff| <= TRANSFER_DUST).
        The identity input returns itself with zero dust.
    """
    if all(c == 0 for c in codes):
        return (((codes, 1.0),), 0.0)
    key = None
    if cache_key is not None:
        key = (cache_key, codes)
        hit = _TRANSFER_CACHE.get(key)
        if hit is not None:
            return hit

    k = len(codes)
    basis, basis_codes = _pauli_basis(k)
    p = local_pauli_matrix(codes)
    m = unitary.conj().T @ p @ unitary
    # d_Q = tr(Q M) / 2^k ; einsum over the stacked basis
    coeffs = np.einsum("qij,ji->q", basis, m) / (2**k)
    out = []
    dust = 0.0
    for c, oc in zip(coeffs, basis_codes):
        if abs(c.imag) > 1e-9:  # would indicate a non-unitary/broken input
            raise AssertionError(
                "Pauli transfer produced a non-Hermitian expansion; "
                "this is a bug or a non-unitary gate matrix"
            )
        if abs(c.real) > TRANSFER_DUST:
            out.append((oc, float(c.real)))
        else:
            dust += abs(float(c.real))
    out.sort(key=lambda t: -abs(t[1]))
    result: Transfer = (tuple(out), dust)
    if key is not None:
        cache_put(_TRANSFER_CACHE, key, result, _TRANSFER_CACHE_LIMIT)
    return result


_DEVIATION_CACHE: dict[tuple, float] = {}


def cached_deviation(
    cache_key: tuple, codes: LocalPauli, transfer: Transfer
) -> float:
    """Memoised :func:`transfer_deviation` keyed like the transfer cache."""
    key = (cache_key, codes)
    hit = _DEVIATION_CACHE.get(key)
    if hit is None:
        hit = transfer_deviation(codes, transfer)
        cache_put(_DEVIATION_CACHE, key, hit, _TRANSFER_CACHE_LIMIT)
    return hit


def transfer_deviation(codes: LocalPauli, transfer: Transfer) -> float:
    """Provable bound on ``||U^dag P U - P||`` from a transfer expansion.

    Equals ``|d_P - 1| + sum_{Q != P} |d_Q| + dust`` (triangle inequality;
    each Pauli has operator norm 1). Zero iff the gate exactly commutes
    with P.
    """
    terms, dust = transfer
    dev = dust
    seen_self = False
    for oc, d in terms:
        if oc == codes:
            dev += abs(d - 1.0)
            seen_self = True
        else:
            dev += abs(d)
    if not seen_self:
        dev += 1.0
    return dev


class GateTable:
    """Per-gate-instance transfer table with precomputed bit patterns.

    For a fixed gate (unitary + concrete qubits), maps each local input
    Pauli code to ``(patterns, dust, deviation)`` where patterns is a tuple
    of ``(x_pattern, z_pattern, coefficient)`` — the output term is
    ``(base_x | x_pattern, base_z | z_pattern)`` with ``base`` the input
    masks cleared on the gate qubits. Purely a memoisation of the exact
    transfer; no arithmetic differs from conjugate_pauli.
    """

    __slots__ = ("unitary", "cache_key", "qubits", "entries", "_vec")

    def __init__(self, unitary, cache_key: tuple, qubits: tuple[int, ...]) -> None:
        self.unitary = unitary
        self.cache_key = cache_key
        self.qubits = qubits
        self.entries: dict[LocalPauli, tuple] = {}
        self._vec: tuple | None = None

    def lookup(self, codes: LocalPauli) -> tuple:
        ent = self.entries.get(codes)
        if ent is None:
            tterms, tdust = conjugate_pauli(self.unitary, codes, self.cache_key)
            dev = transfer_deviation(codes, (tterms, tdust))
            patterns = []
            for out_codes, d in tterms:
                x = z = 0
                for q, code in zip(self.qubits, out_codes):
                    if code == 1:
                        x |= 1 << q
                    elif code == 2:
                        x |= 1 << q
                        z |= 1 << q
                    elif code == 3:
                        z |= 1 << q
                patterns.append((x, z, d))
            ent = (tuple(patterns), tdust, dev)
            self.entries[codes] = ent
        return ent

    def vec_arrays(self) -> tuple:
        """Flat numpy arrays of the full local transfer, for the
        vectorised tracker path.

        Combined-code convention: gate qubit ``j`` (position in
        ``self.qubits``) contributes ``code_j * 4**j`` to the index, with
        ``code_j`` the usual 0..3 = I,X,Y,Z. Patterns are stored flat as
        ``pat_*[offsets[c] : offsets[c] + counts[c]]`` per combined code
        ``c``. Only valid for qubit indices <= 62 (int64 masks) — the
        tracker never promotes otherwise. Purely a re-layout of the same
        exact transfer data; no arithmetic differs.
        """
        if self._vec is None:
            k = len(self.qubits)
            n_codes = 4**k
            px: list[int] = []
            pz: list[int] = []
            pd: list[float] = []
            offsets = np.empty(n_codes, dtype=np.int64)
            counts = np.empty(n_codes, dtype=np.int64)
            dev = np.empty(n_codes, dtype=np.float64)
            tdust = np.empty(n_codes, dtype=np.float64)
            for c in range(n_codes):
                codes = tuple((c >> (2 * j)) & 3 for j in range(k))
                patterns, td, dv = self.lookup(codes)
                offsets[c] = len(px)
                counts[c] = len(patterns)
                dev[c] = dv
                tdust[c] = td
                for x, z, d in patterns:
                    px.append(x)
                    pz.append(z)
                    pd.append(d)
            self._vec = (
                np.array(px, dtype=np.int64),
                np.array(pz, dtype=np.int64),
                np.array(pd, dtype=np.float64),
                offsets,
                counts,
                dev,
                tdust,
            )
        return self._vec


_GATE_TABLE_CACHE: dict[tuple, GateTable] = {}
_GATE_TABLE_LIMIT = 100_000


def gate_table(unitary, cache_key: tuple, qubits: tuple[int, ...]) -> GateTable:
    """Fetch (or create) the memoised transfer table of a gate instance."""
    key = (cache_key, qubits)
    tbl = _GATE_TABLE_CACHE.get(key)
    if tbl is None:
        tbl = GateTable(unitary, cache_key, qubits)
        cache_put(_GATE_TABLE_CACHE, key, tbl, _GATE_TABLE_LIMIT)
    return tbl


def clear_caches() -> None:
    """Drop all memoised transfers (mainly for tests/benchmark hygiene)."""
    _TRANSFER_CACHE.clear()
    _DEVIATION_CACHE.clear()
    _GATE_TABLE_CACHE.clear()


def cache_stats() -> dict[str, int]:
    """Size of the transfer cache (for diagnostics)."""
    return {"transfer_entries": len(_TRANSFER_CACHE)}


def iter_axis_codes(term_x: int, term_z: int, qubits: Iterable[int]) -> LocalPauli:
    """Extract local axis codes of a global (x, z) mask pair on given qubits."""
    out = []
    for q in qubits:
        xb = (term_x >> q) & 1
        zb = (term_z >> q) & 1
        # (x,z) -> code: I=0, X=1, Y=2, Z=3
        out.append((1 if xb and not zb else 2 if xb and zb else 3 if zb else 0))
    return tuple(out)
