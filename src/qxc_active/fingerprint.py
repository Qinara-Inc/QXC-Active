# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Stable fingerprints of what a report certifies.

A certificate that cannot be matched to its subject is not much of a
certificate, so every report carries a ``subject`` block: a stable hash of
the canonicalised input circuit, a hash of each observable, and the
resolved reduction options.

Stability contract: two identical inputs produce identical hashes across
runs, machines and ``PYTHONHASHSEED`` values. Everything here therefore
serialises deterministically — sorted structures, ``repr`` of floats
(shortest round-trip, IEEE-754-determined), raw little-endian bytes for
matrices — and never touches Python's randomised ``hash()``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .ir import Circuit
from .paulis import Distribution, Measurement


def _h() -> "hashlib._Hash":
    return hashlib.sha256()


def circuit_fingerprint(circuit: Circuit) -> str:
    """SHA-256 of the canonical serialisation of a circuit."""
    h = _h()
    h.update(b"qxc-circuit-v1\0")
    h.update(str(circuit.n_qubits).encode())
    for gate in circuit.gates:
        h.update(b"\0g\0")
        h.update(gate.name.encode())
        h.update(("," + ",".join(map(str, gate.qubits))).encode())
        h.update(("," + ",".join(repr(p) for p in gate.params)).encode())
        if gate.matrix is not None:
            m = np.ascontiguousarray(gate.matrix, dtype=np.complex128)
            h.update(str(m.shape).encode())
            h.update(m.tobytes())
    return h.hexdigest()


def observable_fingerprint(measurement: Measurement) -> str:
    """SHA-256 of the mathematical content of one measurement.

    The (cosmetic) name is deliberately excluded: two observables with the
    same terms certify the same thing.
    """
    h = _h()
    if isinstance(measurement, Distribution):
        h.update(b"qxc-distribution-v1\0")
        h.update(",".join(map(str, measurement.qubits)).encode())
    else:
        h.update(b"qxc-observable-v1\0")
        for coeff, term in sorted(
            measurement.terms, key=lambda ct: (ct[1].x, ct[1].z)
        ):
            h.update(f"{term.x},{term.z},{repr(float(coeff))};".encode())
    return h.hexdigest()


def subject_block(
    circuit: Circuit,
    measurements: list[Measurement],
    options: dict,
) -> dict:
    """The report's ``subject``: what exactly this certificate is about."""
    return {
        "circuit_sha256": circuit_fingerprint(circuit),
        "observables_sha256": [
            observable_fingerprint(m) for m in measurements
        ],
        "options": dict(sorted(options.items())),
    }
