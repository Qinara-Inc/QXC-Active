# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Gate library: named gates, their unitaries, and small-matrix utilities.

Every named gate maps to an exact unitary built with numpy. The qubit-order
convention used throughout the package: for a gate on qubits ``(a, b, ...)`` the
matrix acts on basis states ``|q_a q_b ...>`` with ``q_a`` the *most
significant* bit (textbook convention, e.g. CX = diag-block(I, X) with the
control first).

Gates not in the registry can still be used by supplying an explicit unitary
matrix on the :class:`~qxc_active.ir.Gate`.
"""

from __future__ import annotations

import cmath
import math
from typing import Callable, Mapping

import numpy as np

from .caches import cache_put
from .errors import InvalidCircuitError, UnsupportedGateError

Matrix = np.ndarray

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)

PAULI_MATRICES: tuple[Matrix, ...] = (_I2, _X, _Y, _Z)  # codes 0..3 = I,X,Y,Z


def _rx(theta: float) -> Matrix:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> Matrix:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> Matrix:
    return np.array(
        [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]],
        dtype=complex,
    )


def _p(lam: float) -> Matrix:
    return np.array([[1, 0], [0, cmath.exp(1j * lam)]], dtype=complex)


def _u(theta: float, phi: float, lam: float) -> Matrix:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [c, -cmath.exp(1j * lam) * s],
            [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


def _u2(phi: float, lam: float) -> Matrix:
    return _u(math.pi / 2, phi, lam)


def _controlled(u: Matrix) -> Matrix:
    """Add one control qubit (control = most significant)."""
    k = u.shape[0]
    out = np.eye(2 * k, dtype=complex)
    out[k:, k:] = u
    return out


def _rxx(theta: float) -> Matrix:
    c, s = math.cos(theta / 2), -1j * math.sin(theta / 2)
    out = np.eye(4, dtype=complex) * c
    out[0, 3] = out[3, 0] = out[1, 2] = out[2, 1] = s
    return out


def _ryy(theta: float) -> Matrix:
    c, s = math.cos(theta / 2), 1j * math.sin(theta / 2)
    out = np.eye(4, dtype=complex) * c
    out[0, 3] = out[3, 0] = s
    out[1, 2] = out[2, 1] = -s
    return out


def _rzz(theta: float) -> Matrix:
    e_m, e_p = cmath.exp(-1j * theta / 2), cmath.exp(1j * theta / 2)
    return np.diag([e_m, e_p, e_p, e_m]).astype(complex)


_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)
_ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)


class GateSpec:
    """Registry entry: arity, parameter count and unitary constructor."""

    __slots__ = ("name", "n_qubits", "n_params", "builder")

    def __init__(
        self,
        name: str,
        n_qubits: int,
        n_params: int,
        builder: Callable[..., Matrix],
    ) -> None:
        self.name = name
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.builder = builder


REGISTRY: Mapping[str, GateSpec] = {
    spec.name: spec
    for spec in [
        # 1-qubit, no parameters
        GateSpec("id", 1, 0, lambda: _I2),
        GateSpec("x", 1, 0, lambda: _X),
        GateSpec("y", 1, 0, lambda: _Y),
        GateSpec("z", 1, 0, lambda: _Z),
        GateSpec("h", 1, 0, lambda: _H),
        GateSpec("s", 1, 0, lambda: _S),
        GateSpec("sdg", 1, 0, lambda: _S.conj().T),
        GateSpec("t", 1, 0, lambda: _T),
        GateSpec("tdg", 1, 0, lambda: _T.conj().T),
        GateSpec("sx", 1, 0, lambda: _SX),
        GateSpec("sxdg", 1, 0, lambda: _SX.conj().T),
        # 1-qubit, parameterised
        GateSpec("rx", 1, 1, _rx),
        GateSpec("ry", 1, 1, _ry),
        GateSpec("rz", 1, 1, _rz),
        GateSpec("p", 1, 1, _p),
        GateSpec("u1", 1, 1, _p),
        GateSpec("u2", 1, 2, _u2),
        GateSpec("u3", 1, 3, _u),
        GateSpec("u", 1, 3, _u),
        # 2-qubit
        GateSpec("cx", 2, 0, lambda: _controlled(_X)),
        GateSpec("cy", 2, 0, lambda: _controlled(_Y)),
        GateSpec("cz", 2, 0, lambda: _controlled(_Z)),
        GateSpec("ch", 2, 0, lambda: _controlled(_H)),
        GateSpec("csx", 2, 0, lambda: _controlled(_SX)),
        GateSpec("cp", 2, 1, lambda lam: _controlled(_p(lam))),
        GateSpec("cu1", 2, 1, lambda lam: _controlled(_p(lam))),
        GateSpec("crx", 2, 1, lambda th: _controlled(_rx(th))),
        GateSpec("cry", 2, 1, lambda th: _controlled(_ry(th))),
        GateSpec("crz", 2, 1, lambda th: _controlled(_rz(th))),
        GateSpec("cu3", 2, 3, lambda t, p_, l: _controlled(_u(t, p_, l))),
        GateSpec(
            "cu", 2, 4,
            lambda t, p_, l, g: _controlled(cmath.exp(1j * g) * _u(t, p_, l)),
        ),
        GateSpec("rxx", 2, 1, _rxx),
        GateSpec("ryy", 2, 1, _ryy),
        GateSpec("rzz", 2, 1, _rzz),
        GateSpec("swap", 2, 0, lambda: _SWAP),
        GateSpec("iswap", 2, 0, lambda: _ISWAP),
        # 3-qubit
        GateSpec("ccx", 3, 0, lambda: _controlled(_controlled(_X))),
        GateSpec("ccz", 3, 0, lambda: _controlled(_controlled(_Z))),
        GateSpec("cswap", 3, 0, lambda: _controlled(_SWAP)),
    ]
}

# Rotation families whose adjacent same-qubit instances merge by adding angles.
MERGEABLE_ROTATIONS: frozenset[str] = frozenset(
    {"rx", "ry", "rz", "p", "u1", "cp", "cu1", "crx", "cry", "crz", "rxx", "ryy", "rzz"}
)

# 2-qubit gates invariant under swapping their qubit arguments.
SYMMETRIC_GATES: frozenset[str] = frozenset(
    {"cz", "cp", "cu1", "rxx", "ryy", "rzz", "swap", "iswap", "ccz"}
)

_MATRIX_CACHE: dict[tuple, Matrix] = {}


def build_unitary(name: str, params: tuple[float, ...]) -> Matrix:
    """Return the (cached, read-only) unitary of a registered named gate."""
    key = (name, params)
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    spec = REGISTRY.get(name)
    if spec is None:
        raise UnsupportedGateError(
            f"unknown gate {name!r}; register it by passing an explicit "
            f"matrix= to Gate, or use one of: {', '.join(sorted(REGISTRY))}"
        )
    if len(params) != spec.n_params:
        raise InvalidCircuitError(
            f"gate {name!r} takes {spec.n_params} parameter(s), got {len(params)}"
        )
    m = np.asarray(spec.builder(*params), dtype=complex)
    m.setflags(write=False)
    cache_put(_MATRIX_CACHE, key, m, 200_000)
    return m


def is_unitary(m: Matrix, tol: float = 1e-10) -> bool:
    """Check unitarity of a small square matrix."""
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    d = m.shape[0]
    return bool(np.allclose(m.conj().T @ m, np.eye(d), atol=tol))


def deviation_from_phase_identity(m: Matrix) -> tuple[float, complex]:
    """Distance of a unitary from ``e^{i phi} * I``.

    Returns ``(dev, phase)`` where ``dev = || m - phase*I ||_2`` (spectral
    norm) minimized over the phase choice used (phase of the trace). For a
    matrix nowhere near a phase multiple of identity, dev is simply large;
    callers compare it against a tolerance.
    """
    tr = np.trace(m)
    if abs(tr) < 1e-14:
        return float(np.linalg.norm(m - np.eye(m.shape[0]), 2)), 1.0 + 0j
    phase = tr / abs(tr)
    dev = float(np.linalg.norm(m - phase * np.eye(m.shape[0]), 2))
    return dev, complex(phase)


def embed(u: Matrix, gate_qubits: tuple[int, ...], union_qubits: tuple[int, ...]) -> Matrix:
    """Embed a gate unitary into a larger qubit set (identity elsewhere).

    ``union_qubits`` must contain all of ``gate_qubits``; the returned matrix
    acts on ``union_qubits`` with the same MSB-first ordering convention.
    """
    extra = [q for q in union_qubits if q not in gate_qubits]
    order = list(gate_qubits) + extra
    k = len(union_qubits)
    full = np.kron(u, np.eye(2 ** len(extra), dtype=complex))
    t = full.reshape((2,) * (2 * k))
    perm = [order.index(q) for q in union_qubits]
    t = np.transpose(t, perm + [k + p for p in perm])
    return np.ascontiguousarray(t.reshape(2**k, 2**k))
