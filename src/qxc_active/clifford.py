# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Targeted Clifford resynthesis: replace a Clifford circuit prefix with a
freshly synthesized preparation of the same stabilizer state.

The key soundness idea: a Clifford block's *unitary* need not be
preserved — only the *state* it produces from |0...0>, because every
measured expectation depends on that state alone. A Clifford prefix
produces a stabilizer state, which a tableau tracks exactly in integer
arithmetic; synthesizing a fresh preparation circuit for the same state is
an exact, zero-error rewrite.

Safety properties:

* All tableau updates route through the same numerically-exact Pauli
  transfer engine used (and tested) everywhere else; coefficients are
  asserted to be exactly +/-1.
* The synthesis ends with a hard assertion that every stabilizer generator
  equals +Z_q. Any surprise aborts the resynthesis (returns None) — the
  caller keeps the original circuit. A bug here can cost an optimization,
  never correctness.
* The caller replaces the prefix only if the new circuit is strictly
  smaller (fewer gates).
"""

from __future__ import annotations

import numpy as np

from .errors import ActiveError
from .gates import PAULI_MATRICES
from .ir import Gate
from .transfer import conjugate_pauli, iter_axis_codes


class CliffordEngineError(ActiveError):
    """The tableau engine hit a state it cannot handle exactly.

    Raised unconditionally (never via ``assert``, so it survives ``-O``)
    because the resynthesis rewrite is charged zero error and is only sound
    if every tableau update is exact. Callers treat it as "abort the
    resynthesis, keep the original circuit" — a lost optimization, never a
    lost guarantee.
    """

# Gates with tableau semantics here (all Clifford, no parameters). A gate
# outside this set (or any raw-matrix gate) ends the resynthesizable prefix.
CLIFFORD_NAMES = frozenset(
    {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "cy", "swap", "sx", "sxdg"}
)

_INVERSE = {
    "h": "h", "s": "sdg", "sdg": "s", "x": "x", "y": "y", "z": "z",
    "cx": "cx", "cz": "cz", "cy": "cy", "swap": "swap",
    "sx": "sxdg", "sxdg": "sx",
}

# single-qubit Pauli multiplication table: (a, b) -> (c, i-power) with
# P_a @ P_b = i^power * P_c ; built numerically once, exactly.
_PROD: dict[tuple[int, int], tuple[int, int]] = {}


def _build_prod_table() -> None:
    if _PROD:
        return
    for a in range(4):
        for b in range(4):
            m = PAULI_MATRICES[a] @ PAULI_MATRICES[b]
            for c in range(4):
                for e in range(4):
                    if np.allclose(m, (1j**e) * PAULI_MATRICES[c], atol=1e-12):
                        _PROD[(a, b)] = (c, e)
                        break
                else:
                    continue
                break
    if len(_PROD) != 16:
        raise CliffordEngineError("Pauli product table incomplete")


def _code(x: int, z: int, q: int) -> int:
    xb, zb = (x >> q) & 1, (z >> q) & 1
    return 1 if xb and not zb else 2 if xb and zb else 3 if zb else 0


class Tableau:
    """Stabilizer generators of a state, as (x mask, z mask, sign) rows."""

    __slots__ = ("n", "xs", "zs", "signs")

    def __init__(self, n: int) -> None:
        self.n = n
        self.xs = [0] * n
        self.zs = [1 << q for q in range(n)]  # |0...0> is stabilized by +Z_q
        self.signs = [1] * n

    def apply_gate(self, gate: Gate) -> None:
        """Conjugate every generator through a Clifford gate (exactly).

        Stabilizers evolve FORWARD: g -> U g U^dag. The transfer engine
        computes U^dag P U, so it is fed the inverse (adjoint) unitary.
        """
        unitary = np.ascontiguousarray(gate.unitary().conj().T)
        ckey = ("dag", gate.cache_key())
        for i in range(self.n):
            codes = iter_axis_codes(self.xs[i], self.zs[i], gate.qubits)
            if all(c == 0 for c in codes):
                continue
            terms, dust = conjugate_pauli(unitary, codes, ckey)
            # a Clifford maps a Pauli to exactly one signed Pauli; float
            # noise (~1e-16 trace dust) is representational only because the
            # sign below is checked to be exactly +/-1. These checks are
            # unconditional raises (not asserts): the rewrite is charged
            # zero error, which is only sound if the tableau is exact.
            if len(terms) != 1 or dust >= 1e-9:
                raise CliffordEngineError(
                    f"gate {gate.name} did not act as a Clifford"
                )
            out_codes, d = terms[0]
            sign = round(d)
            if abs(d - sign) >= 1e-9 or sign not in (1, -1):
                raise CliffordEngineError(
                    f"gate {gate.name}: non-unit Pauli coefficient {d!r}"
                )
            x, z = self.xs[i], self.zs[i]
            for q in gate.qubits:
                mask = ~(1 << q)
                x &= mask
                z &= mask
            for q, code in zip(gate.qubits, out_codes):
                if code in (1, 2):
                    x |= 1 << q
                if code in (2, 3):
                    z |= 1 << q
            self.xs[i], self.zs[i], self.signs[i] = x, z, sign * self.signs[i]

    def row_mult(self, i: int, j: int) -> None:
        """Generator i *= generator j (a free group operation)."""
        _build_prod_table()
        exp = 0
        x, z = 0, 0
        support = self.xs[i] | self.zs[i] | self.xs[j] | self.zs[j]
        q = 0
        m = support
        while m:
            if m & 1:
                c, e = _PROD[
                    (_code(self.xs[i], self.zs[i], q),
                     _code(self.xs[j], self.zs[j], q))
                ]
                exp += e
                if c in (1, 2):
                    x |= 1 << q
                if c in (2, 3):
                    z |= 1 << q
            m >>= 1
            q += 1
        if exp % 2 != 0:
            raise CliffordEngineError(
                "product of commuting generators must be real"
            )
        phase = 1 if exp % 4 == 0 else -1
        self.xs[i], self.zs[i] = x, z
        self.signs[i] = self.signs[i] * self.signs[j] * phase


def simulate_prefix(n_qubits: int, gates: list[Gate]) -> Tableau:
    """Tableau of the state produced by a list of Clifford gates on |0..0>."""
    tab = Tableau(n_qubits)
    for g in gates:
        tab.apply_gate(g)
    return tab


def _row_reduce_x(tab: Tableau) -> list[int]:
    """Reduced row echelon of the X block via free row ops; returns pivots."""
    pivots: list[int] = []
    row = 0
    for col in range(tab.n):
        pivot = next(
            (r for r in range(row, tab.n) if (tab.xs[r] >> col) & 1), None
        )
        if pivot is None:
            continue
        if pivot != row:
            for attr in ("xs", "zs", "signs"):
                lst = getattr(tab, attr)
                lst[row], lst[pivot] = lst[pivot], lst[row]
        for r in range(tab.n):
            if r != row and (tab.xs[r] >> col) & 1:
                tab.row_mult(r, row)
        pivots.append(col)
        row += 1
    return pivots


def synthesize_preparation(tab: Tableau) -> list[Gate] | None:
    """A fresh circuit preparing the tableau's state from |0...0>.

    Returns None if any internal sanity assertion would fail (caller then
    keeps the original circuit). The returned circuit is exact: it prepares
    a state with the identical stabilizer group including signs.
    """
    n = tab.n
    work = Tableau(n)
    work.xs = list(tab.xs)
    work.zs = list(tab.zs)
    work.signs = list(tab.signs)
    ops: list[Gate] = []

    def do(name: str, qubits: tuple[int, ...]) -> None:
        g = Gate(name, qubits)
        work.apply_gate(g)
        ops.append(g)

    try:
        # 1) make the X block full rank: repeatedly row-reduce and give the
        #    first Z-only generator an H on a column outside the current
        #    pivot set (standard-form property guarantees one exists and
        #    that the rank grows; the guard makes any surprise a safe bail)
        guard = 0
        while True:
            pivots = _row_reduce_x(work)
            if len(pivots) == n:
                break
            guard += 1
            if guard > 2 * n + 4:
                return None
            r = len(pivots)  # first Z-only row after reduction
            assert work.xs[r] == 0
            pivot_set = set(pivots)
            free = next(
                (
                    q
                    for q in range(n)
                    if (work.zs[r] >> q) & 1 and q not in pivot_set
                ),
                None,
            )
            if free is None:
                return None  # unexpected structure; bail out safely
            do("h", (free,))

        # 2) Gauss-Jordan the X block to a permutation matrix using CX
        for r in range(n):
            p = pivots[r]
            extra = [
                q for q in range(n) if q != p and (work.xs[r] >> q) & 1
            ]
            for q in extra:
                do("cx", (p, q))
        # rows now have a single X bit each, at their pivot column

        # 3) clear the Z block: S for diagonal entries, CZ for pairs
        for r in range(n):
            p = pivots[r]
            if (work.zs[r] >> p) & 1:
                do("s", (p,))
                if (work.zs[r] >> p) & 1:  # engine says S toggles Y->X path
                    do("s", (p,))
        for r in range(n):
            p = pivots[r]
            for q in range(n):
                if q != p and (work.zs[r] >> q) & 1:
                    do("cz", (p, q))
                    assert not (work.zs[r] >> q) & 1

        # 4) rotate X_p -> Z_p and fix signs
        for q in range(n):
            do("h", (q,))
        for r in range(n):
            if work.signs[r] == -1:
                do("x", (pivots[r],))

        # final hard check: generators must be exactly {+Z_q}
        got = sorted(
            (work.xs[r], work.zs[r], work.signs[r]) for r in range(n)
        )
        want = sorted((0, 1 << q, 1) for q in range(n))
        if got != want:
            return None
    except (AssertionError, CliffordEngineError):
        return None

    # ops maps the target state to |0..0>; the preparation is the reverse
    # with each gate inverted
    return [
        Gate(_INVERSE[g.name], g.qubits) for g in reversed(ops)
    ]


def clifford_prefix_length(gates: list[Gate]) -> int:
    """Length of the maximal all-Clifford prefix (cheap name check)."""
    k = 0
    for g in gates:
        if g.matrix is not None or g.name not in CLIFFORD_NAMES:
            break
        k += 1
    return k
