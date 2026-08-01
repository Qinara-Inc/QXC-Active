# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""qxc_active — the ACTIVE primitive: observable-aware circuit reduction.

Reduces a quantum circuit to only what affects the measured observables —
exactly by default, or under a proven per-observable error bound.

Quick start::

    from qxc_active import Circuit, active_reduce

    circ = Circuit(3).add("h", 0).add("cx", (0, 1)).add("cx", (1, 2))
    reduced, report = active_reduce(circ, ["Z0"])
    print(report.summary())
"""

from __future__ import annotations

# Single source of truth for the package version: pyproject.toml reads it
# from here (setuptools dynamic version). It appears in every emitted
# report, so a mismatch would mislabel every artifact.
__version__ = "0.3.0"

from .errors import (  # noqa: E402
    ActiveError,
    BudgetViolationError,
    InvalidCircuitError,
    InvalidObservableError,
    InvalidOptionError,
    QasmExportError,
    QasmImportError,
    UnsupportedGateError,
)
from .ir import Circuit, Gate  # noqa: E402
from .paulis import Distribution, Observable, PauliTerm  # noqa: E402
from .reduce import active_reduce  # noqa: E402
from .report import Report, RemovedGate  # noqa: E402

__all__ = [
    "__version__",
    "ActiveError",
    "BudgetViolationError",
    "Circuit",
    "Distribution",
    "Gate",
    "InvalidCircuitError",
    "InvalidObservableError",
    "InvalidOptionError",
    "Observable",
    "PauliTerm",
    "QasmExportError",
    "QasmImportError",
    "RemovedGate",
    "Report",
    "UnsupportedGateError",
    "active_reduce",
    "convert_qasm2_to_qasm3",
    "load_qasm",
    "loads_qasm",
    "dump_qasm",
    "dumps_qasm",
    "measured_qubits",
    "verify_reduction",
    "VerificationError",
    "VerificationResult",
]


def __getattr__(name: str):
    # QASM I/O and the brute-force verifier are imported lazily: the core
    # algorithm has no import-time dependency on the parser package, and the
    # default reduce path never even loads the verifier module.
    if name in (
        "load_qasm",
        "loads_qasm",
        "dump_qasm",
        "dumps_qasm",
        "measured_qubits",
        "convert_qasm2_to_qasm3",
    ):
        from . import qasm

        return getattr(qasm, name)
    if name in ("verify_reduction", "VerificationError", "VerificationResult"):
        from . import verifier

        return getattr(verifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
