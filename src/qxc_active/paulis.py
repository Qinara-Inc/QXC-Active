# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Pauli strings and Hermitian observables.

A Pauli string on ``n`` qubits is stored as a pair of Python integers
``(x, z)`` used as bit masks: bit ``q`` of ``x`` set means an X component on
qubit ``q``, bit ``q`` of ``z`` a Z component; both bits set means Y. All
operators here are the standard *Hermitian* Paulis (Y = [[0,-i],[i,0]]), so a
real-weighted sum of them is a Hermitian observable and stays real-weighted
under conjugation by any unitary.

Text formats accepted by :func:`PauliTerm.parse` / :func:`Observable.parse`:

* sparse:  ``"Z0"``, ``"X1*Z4"``, ``"Y2 Z3"`` (``*`` or whitespace separated)
* dense:   ``"IZXY"`` — qubit 0 is the leftmost character
* sums:    ``"0.5*Z0*Z1 + 0.25*X2 - Y0"`` (Observable only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Union

from .errors import InvalidObservableError

_AXES = "IXYZ"
# axis code -> (x bit, z bit)
_AXIS_BITS = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_BITS_AXIS = {v: k for k, v in _AXIS_BITS.items()}

_SPARSE_FACTOR = re.compile(r"^([IXYZ])(\d+)$")


def support_bits(mask: int) -> Iterator[int]:
    """Yield the set bit positions of ``mask`` in ascending order."""
    q = 0
    while mask:
        if mask & 1:
            yield q
        mask >>= 1
        q += 1


@dataclass(frozen=True)
class PauliTerm:
    """A single Pauli string (no coefficient). Immutable and hashable.

    Attributes:
        x: X-component bit mask (bit q set = X or Y on qubit q).
        z: Z-component bit mask (bit q set = Z or Y on qubit q).
    """

    x: int
    z: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.z < 0:
            raise InvalidObservableError("Pauli bit masks must be non-negative")

    @property
    def support(self) -> int:
        """Bit mask of qubits on which this Pauli acts non-trivially."""
        return self.x | self.z

    def support_qubits(self) -> list[int]:
        """Sorted list of qubits in the support."""
        return list(support_bits(self.support))

    def axis(self, qubit: int) -> str:
        """Return 'I', 'X', 'Y' or 'Z' for the given qubit."""
        return _BITS_AXIS[((self.x >> qubit) & 1, (self.z >> qubit) & 1)]

    def is_identity(self) -> bool:
        """True if this term is the identity operator."""
        return self.x == 0 and self.z == 0

    def max_qubit(self) -> int:
        """Highest qubit index in the support; -1 for identity."""
        return self.support.bit_length() - 1

    @staticmethod
    def from_axes(axes: dict[int, str]) -> "PauliTerm":
        """Build from a {qubit: 'X'|'Y'|'Z'|'I'} mapping."""
        x = z = 0
        for q, a in axes.items():
            if q < 0:
                raise InvalidObservableError(f"negative qubit index {q}")
            if a not in _AXIS_BITS:
                raise InvalidObservableError(f"invalid Pauli axis {a!r}")
            xb, zb = _AXIS_BITS[a]
            x |= xb << q
            z |= zb << q
        return PauliTerm(x, z)

    @staticmethod
    def parse(text: str) -> "PauliTerm":
        """Parse sparse (``"X1*Z4"``) or dense (``"IZXY"``) notation."""
        s = text.strip()
        if not s:
            raise InvalidObservableError("empty Pauli string")
        if s.startswith("*") or s.endswith("*"):
            raise InvalidObservableError(
                f"dangling '*' in Pauli string {text!r}"
            )
        if re.fullmatch(r"[IXYZ]+", s) and not _SPARSE_FACTOR.match(s):
            # dense: leftmost char is qubit 0
            return PauliTerm.from_axes({q: a for q, a in enumerate(s)})
        axes: dict[int, str] = {}
        for factor in re.split(r"[*\s]+", s):
            if not factor:
                continue
            m = _SPARSE_FACTOR.match(factor)
            if not m:
                raise InvalidObservableError(
                    f"cannot parse Pauli factor {factor!r} in {text!r} "
                    "(expected e.g. 'Z0', 'X3', or a dense string like 'IZZ')"
                )
            a, q = m.group(1), int(m.group(2))
            if q in axes:
                raise InvalidObservableError(
                    f"qubit {q} appears twice in Pauli string {text!r}"
                )
            axes[q] = a
        return PauliTerm.from_axes(axes)

    def to_sparse(self) -> str:
        """Sparse text form, e.g. ``"X1*Z4"``; ``"I"`` for identity."""
        if self.is_identity():
            return "I"
        return "*".join(f"{self.axis(q)}{q}" for q in self.support_qubits())

    def to_dense(self, n_qubits: int) -> str:
        """Dense text form on ``n_qubits`` qubits (qubit 0 leftmost)."""
        if self.max_qubit() >= n_qubits:
            raise InvalidObservableError(
                f"Pauli {self.to_sparse()} acts on qubit {self.max_qubit()} "
                f"but only {n_qubits} qubits exist"
            )
        return "".join(self.axis(q) for q in range(n_qubits))

    def __str__(self) -> str:
        return self.to_sparse()


class Distribution:
    """A measurement of the full joint outcome distribution on some qubits.

    Passing ``Distribution((0, 1, 2))`` to ``active_reduce`` asks it to
    preserve the complete sampling distribution of a computational-basis
    measurement of those qubits (not just individual expectations).

    Guarantee semantics: in exact mode the distribution is preserved (up to
    reported float dust); with ``epsilon > 0`` every individual outcome
    probability is guaranteed to change by at most the reported certified
    bound, which never exceeds epsilon. (This is a per-outcome bound, NOT a
    total-variation bound — a TV guarantee would carry a 2^|qubits| factor
    and is deliberately not claimed.)
    """

    __slots__ = ("qubits", "name")

    def __init__(self, qubits, name: str | None = None) -> None:
        qs = tuple(int(q) for q in qubits)
        if not qs:
            raise InvalidObservableError("Distribution needs at least one qubit")
        if len(set(qs)) != len(qs):
            raise InvalidObservableError(
                f"Distribution lists a qubit twice: {qs}"
            )
        if any(q < 0 for q in qs):
            raise InvalidObservableError(
                f"Distribution has a negative qubit index: {qs}"
            )
        self.qubits = tuple(sorted(qs))
        self.name = name

    def max_qubit(self) -> int:
        return max(self.qubits)

    def support(self) -> int:
        m = 0
        for q in self.qubits:
            m |= 1 << q
        return m

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        return f"<Distribution{label} qubits={self.qubits}>"

    def __str__(self) -> str:
        return f"dist({','.join(map(str, self.qubits))})"


ObservableLike = Union["Observable", PauliTerm, str, Distribution]
Measurement = Union["Observable", Distribution]


class Observable:
    """A Hermitian observable: a real-weighted sum of Pauli strings.

    Terms with the same Pauli string are merged; zero-coefficient terms are
    dropped. The empty sum (numerically zero operator) is allowed.
    """

    __slots__ = ("terms", "name")

    def __init__(
        self,
        terms: Iterable[tuple[float, PauliTerm]],
        name: str | None = None,
    ) -> None:
        merged: dict[PauliTerm, float] = {}
        for coeff, term in terms:
            c = float(coeff)
            if not isinstance(term, PauliTerm):
                raise InvalidObservableError(
                    f"expected PauliTerm, got {type(term).__name__}"
                )
            if c != c or c in (float("inf"), float("-inf")):
                raise InvalidObservableError("observable coefficients must be finite")
            merged[term] = merged.get(term, 0.0) + c
        self.terms: tuple[tuple[float, PauliTerm], ...] = tuple(
            (c, t) for t, c in merged.items() if c != 0.0
        )
        self.name = name

    @staticmethod
    def coerce(obj: ObservableLike, name: str | None = None) -> "Observable":
        """Convert a string, PauliTerm or Observable into an Observable."""
        if isinstance(obj, Observable):
            return obj
        if isinstance(obj, PauliTerm):
            return Observable([(1.0, obj)], name=name)
        if isinstance(obj, str):
            return Observable.parse(obj, name=name)
        raise InvalidObservableError(
            f"cannot interpret {type(obj).__name__} as an observable"
        )

    @staticmethod
    def parse(text: str, name: str | None = None) -> "Observable":
        """Parse a sum like ``"0.5*Z0*Z1 + 0.25*X2 - Y0"`` (or one Pauli)."""
        s = text.strip()
        if not s:
            raise InvalidObservableError("empty observable string")
        # split into signed chunks at top-level +/-
        chunks: list[tuple[float, str]] = []
        sign, buf = 1.0, []
        for ch in s:
            if ch in "+-" and buf and buf[-1] not in "eE*":
                chunk = "".join(buf).strip()
                if chunk:
                    chunks.append((sign, chunk))
                sign, buf = (1.0 if ch == "+" else -1.0), []
            elif ch in "+-" and not any(c.strip() for c in buf):
                sign *= 1.0 if ch == "+" else -1.0
            else:
                buf.append(ch)
        chunk = "".join(buf).strip()
        if chunk:
            chunks.append((sign, chunk))
        if not chunks:
            raise InvalidObservableError(f"cannot parse observable {text!r}")

        terms: list[tuple[float, PauliTerm]] = []
        for sgn, body in chunks:
            coeff = sgn
            factors = [f for f in re.split(r"\*", body) if f.strip()]
            pauli_factors: list[str] = []
            for f in factors:
                f = f.strip()
                try:
                    coeff *= float(f)
                    continue
                except ValueError:
                    pauli_factors.append(f)
            if not pauli_factors:
                raise InvalidObservableError(
                    f"term {body!r} in {text!r} has no Pauli factor"
                )
            terms.append((coeff, PauliTerm.parse("*".join(pauli_factors))))
        return Observable(terms, name=name)

    def to_dict(self) -> dict:
        """Structured, JSON-safe form: name, canonical string, term list.

        Each term is ``{"coefficient": float, "paulis": {qubit: axis}}``
        with qubit keys as strings (JSON objects cannot carry int keys).
        Round-trips exactly through :meth:`from_dict`.
        """
        return {
            "name": self.name,
            "string": str(self),
            "terms": [
                {
                    "coefficient": c,
                    "paulis": {
                        str(q): t.axis(q) for q in t.support_qubits()
                    },
                }
                for c, t in self.terms
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "Observable":
        """Inverse of :meth:`to_dict` (accepts int or str qubit keys)."""
        try:
            terms = [
                (
                    float(entry["coefficient"]),
                    PauliTerm.from_axes(
                        {int(q): a for q, a in entry["paulis"].items()}
                    ),
                )
                for entry in data["terms"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidObservableError(
                f"malformed observable dict: {exc}"
            ) from exc
        return Observable(terms, name=data.get("name"))

    def max_qubit(self) -> int:
        """Highest qubit index used by any term; -1 if none."""
        return max((t.max_qubit() for _, t in self.terms), default=-1)

    def support(self) -> int:
        """Union of all terms' supports as a bit mask."""
        m = 0
        for _, t in self.terms:
            m |= t.support
        return m

    def norm_bound(self) -> float:
        """A provable upper bound on the operator norm: sum of |coefficients|."""
        return sum(abs(c) for c, _ in self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for c, t in self.terms:
            parts.append(f"{'+' if c >= 0 else '-'} {abs(c):g}*{t.to_sparse()}")
        out = " ".join(parts)
        return out[2:] if out.startswith("+ ") else out

    def __repr__(self) -> str:
        label = f" name={self.name!r}" if self.name else ""
        return f"<Observable{label} {self}>"


def coerce_observables(
    observables: Sequence[ObservableLike] | ObservableLike,
) -> list[Measurement]:
    """Normalize user input to a non-empty list of measurements.

    Each element becomes an :class:`Observable` (from a string, PauliTerm
    or Observable) or stays a :class:`Distribution`.
    """
    if isinstance(observables, (Observable, PauliTerm, str, Distribution)):
        observables = [observables]
    result: list[Measurement] = []
    for i, o in enumerate(observables):
        if isinstance(o, Distribution):
            if o.name is None:
                o = Distribution(o.qubits, name=str(o))
            result.append(o)
        else:
            result.append(
                Observable.coerce(o, name=getattr(o, "name", None) or f"obs{i}")
            )
    if not result:
        raise InvalidObservableError("at least one observable is required")
    return result
