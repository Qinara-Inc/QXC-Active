# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Circuit intermediate representation.

The IR is deliberately minimal and explicit: a qubit count and an ordered
list of gates, each gate being ``(name, qubits, params)`` with an optional
explicit unitary matrix for gates outside the built-in registry. Measured
observables are *not* part of the circuit; they are passed separately to
:func:`qxc_active.reduce.active_reduce`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from . import gates as _gates
from .errors import InvalidCircuitError


@dataclass(frozen=True)
class Gate:
    """One gate application.

    Attributes:
        name: Gate name. Either a registry name (``"cx"``, ``"rz"``, ...) or
            any label if ``matrix`` is supplied.
        qubits: Target qubits, order-significant (first = most significant
            bit of the matrix, e.g. the control of ``cx``).
        params: Real parameters (rotation angles in radians).
        matrix: Optional explicit unitary (2^k x 2^k for k qubits). Required
            for names outside the registry; if given for a registry name it
            takes precedence.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    matrix: np.ndarray | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))
        if len(set(self.qubits)) != len(self.qubits):
            raise InvalidCircuitError(
                f"gate {self.name!r} lists a qubit twice: {self.qubits}"
            )
        if not self.qubits:
            raise InvalidCircuitError(f"gate {self.name!r} has no target qubits")
        if any(q < 0 for q in self.qubits):
            raise InvalidCircuitError(
                f"gate {self.name!r} has a negative qubit index: {self.qubits}"
            )
        for p in self.params:
            if p != p or p in (float("inf"), float("-inf")):
                raise InvalidCircuitError(
                    f"gate {self.name!r} has a non-finite parameter"
                )
        if self.matrix is not None:
            m = np.asarray(self.matrix, dtype=complex)
            dim = 2 ** len(self.qubits)
            if m.shape != (dim, dim):
                raise InvalidCircuitError(
                    f"gate {self.name!r} on {len(self.qubits)} qubit(s) needs a "
                    f"{dim}x{dim} matrix, got {m.shape}"
                )
            if not _gates.is_unitary(m):
                raise InvalidCircuitError(f"matrix for gate {self.name!r} is not unitary")
            m = m.copy()
            m.setflags(write=False)
            object.__setattr__(self, "matrix", m)
        elif self.name not in _gates.REGISTRY:
            raise InvalidCircuitError(
                f"gate {self.name!r} is not a built-in gate and no matrix was given"
            )
        else:
            spec = _gates.REGISTRY[self.name]
            if len(self.qubits) != spec.n_qubits:
                raise InvalidCircuitError(
                    f"gate {self.name!r} acts on {spec.n_qubits} qubit(s), "
                    f"got {len(self.qubits)}"
                )
            if len(self.params) != spec.n_params:
                raise InvalidCircuitError(
                    f"gate {self.name!r} takes {spec.n_params} parameter(s), "
                    f"got {len(self.params)}"
                )

    def unitary(self) -> np.ndarray:
        """The gate's unitary matrix (read-only ndarray)."""
        if self.matrix is not None:
            return self.matrix
        return _gates.build_unitary(self.name, self.params)

    def cache_key(self) -> tuple:
        """Hashable key identifying the unitary (used by transfer caches)."""
        if self.matrix is not None:
            return ("__matrix__", self.matrix.tobytes(), self.matrix.shape[0])
        return (self.name, self.params)

    def __str__(self) -> str:
        ps = f"({', '.join(f'{p:g}' for p in self.params)})" if self.params else ""
        return f"{self.name}{ps} {', '.join(map(str, self.qubits))}"


class Circuit:
    """An ordered list of gates on ``n_qubits`` qubits (initial state |0...0>).

    ``measured_qubits`` records which qubits the source program measures
    (set by the QASM loaders, default empty). It is purely informational
    — used only to infer a default observable when the caller supplies
    none — and plays no part in reduction, equality, hashing or the
    circuit fingerprint.
    """

    __slots__ = ("n_qubits", "gates", "measured_qubits")

    def __init__(self, n_qubits: int, gates: Iterable[Gate] = ()) -> None:
        n = int(n_qubits)
        if n < 0:
            raise InvalidCircuitError(f"n_qubits must be >= 0, got {n_qubits}")
        self.n_qubits = n
        self.gates: list[Gate] = []
        self.measured_qubits: tuple[int, ...] = ()
        for g in gates:
            self.append(g)

    def append(self, gate: Gate) -> None:
        """Append a gate, validating its qubit indices against n_qubits."""
        if not isinstance(gate, Gate):
            raise InvalidCircuitError(f"expected Gate, got {type(gate).__name__}")
        if max(gate.qubits) >= self.n_qubits:
            raise InvalidCircuitError(
                f"gate {gate} uses qubit {max(gate.qubits)} but the circuit "
                f"has only {self.n_qubits} qubit(s)"
            )
        self.gates.append(gate)

    def add(
        self,
        name: str,
        qubits: Sequence[int] | int,
        params: Sequence[float] | float = (),
        matrix: np.ndarray | None = None,
    ) -> "Circuit":
        """Convenience builder: ``circ.add("cx", (0, 1)).add("rz", 2, 0.3)``."""
        if isinstance(qubits, int):
            qubits = (qubits,)
        if isinstance(params, (int, float)):
            params = (params,)
        self.append(Gate(name, tuple(qubits), tuple(params), matrix))
        return self

    def copy(self) -> "Circuit":
        """Shallow copy (gates are immutable)."""
        c = Circuit(self.n_qubits)
        c.gates = list(self.gates)
        c.measured_qubits = self.measured_qubits
        return c

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    def depth(self) -> int:
        """Circuit depth: longest chain of gates sharing qubits."""
        level: dict[int, int] = {}
        depth = 0
        for g in self.gates:
            d = 1 + max((level.get(q, 0) for q in g.qubits), default=0)
            for q in g.qubits:
                level[q] = d
            depth = max(depth, d)
        return depth

    def two_qubit_count(self) -> int:
        """Number of gates acting on two or more qubits."""
        return sum(1 for g in self.gates if len(g.qubits) >= 2)

    def used_qubits(self) -> set[int]:
        """Qubits touched by at least one gate."""
        out: set[int] = set()
        for g in self.gates:
            out.update(g.qubits)
        return out

    def __len__(self) -> int:
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        if self.n_qubits != other.n_qubits or len(self.gates) != len(other.gates):
            return False
        for a, b in zip(self.gates, other.gates):
            if a.name != b.name or a.qubits != b.qubits or a.params != b.params:
                return False
            if (a.matrix is None) != (b.matrix is None):
                return False
            if a.matrix is not None and not np.array_equal(a.matrix, b.matrix):
                return False
        return True

    def __str__(self) -> str:
        head = f"Circuit({self.n_qubits} qubits, {len(self.gates)} gates)"
        if len(self.gates) <= 20:
            return head + "".join(f"\n  {g}" for g in self.gates)
        return head

    __repr__ = __str__
