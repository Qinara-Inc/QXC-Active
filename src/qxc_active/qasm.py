# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""OpenQASM 3 and OpenQASM 2.0 import; OpenQASM 3 export.

Import covers the practical subset of compiled circuits: qubit
declarations (multiple registers), all ``stdgates.inc`` gates, custom
``gate`` definitions (inlined recursively with parameter substitution),
register broadcast, constant declarations, arithmetic parameter
expressions (``pi``/``tau``/``euler``). Measurements, barriers,
classical declarations and global phases are ignored — none affects
observable reduction. Anything else raises :class:`QasmImportError`
naming the construct; valid QASM never crashes the importer.

OpenQASM 2.0 loads directly: the dialect is auto-detected from the
``OPENQASM`` header (override with ``dialect="qasm2" | "qasm3"``), and
2.0 input goes through a strict, line-preserving source transform
(header, ``qelib1.inc``, ``qreg``/``creg``, ``measure a -> b;``, plus
canonical definitions for the qelib1 gates missing from stdgates: u0,
rccx, rc3x, c3x, c3sqrtx, c4x) into the SAME parser as 3.0 input, so a
2.0 file and its 3.0 equivalent produce the identical circuit — a gate
is never silently rewritten. Constructs the reducer cannot honour
(``opaque``, ``reset``, ``if(...)``) fail with an error naming the
construct and its source line. :func:`measured_qubits` reads a file's
measured qubits (either dialect).

Export writes OpenQASM 3.0 against ``stdgates.inc``; registry gates
missing from stdgates (rxx, ryy, rzz, iswap, sxdg, csx, ccz) are
emitted as local gate definitions verified by the test suite. Raw-
matrix gates have no QASM spelling and raise :class:`QasmExportError`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import openqasm3
from openqasm3 import ast

from .errors import QasmExportError, QasmImportError
from .gates import REGISTRY
from .ir import Circuit, Gate

_CONSTANTS = {
    "pi": math.pi,
    "π": math.pi,
    "tau": 2 * math.pi,
    "τ": 2 * math.pi,
    "euler": math.e,
    "ℇ": math.e,
}

_NAME_ALIASES = {
    "U": "u",
    "CX": "cx",
    "phase": "p",
    "cphase": "cp",
    "i": "id",
    "toffoli": "ccx",
    "fredkin": "cswap",
}

_MAX_INLINE_DEPTH = 200


@dataclass
class _GateDef:
    params: list[str]
    qubit_args: list[str]
    body: list[ast.Statement]


class _Importer:
    def __init__(self, source: str = "", read_as_qasm2: bool = False) -> None:
        self.qubits: dict[str, tuple[int, int]] = {}  # register -> (offset, size)
        self.n_qubits = 0
        self.consts: dict[str, float] = {}
        self.defs: dict[str, _GateDef] = {}
        self.gates: list[Gate] = []
        self.measured: list[int] = []  # qubits measured, in program order
        # original source text (line numbers are 1:1 even for converted
        # 2.0 input) and whether it was read as OpenQASM 2.0 — both used
        # only to make rejection messages actionable
        self.source = source
        self.read_as_qasm2 = read_as_qasm2

    def _where(self, stmt: ast.Statement) -> str:
        line_no = getattr(getattr(stmt, "span", None), "start_line", None)
        ctx = _source_line(self.source, line_no)
        return f" ({ctx})" if ctx else ""

    def _dialect_note(self) -> str:
        return "\n(The file was read as OpenQASM 2.0.)" if self.read_as_qasm2 else ""

    # -- expressions --------------------------------------------------------

    def eval_expr(self, node: ast.Expression, env: dict[str, float]) -> float:
        if isinstance(node, ast.IntegerLiteral):
            return float(node.value)
        if isinstance(node, ast.FloatLiteral):
            return float(node.value)
        if isinstance(node, ast.Identifier):
            name = node.name
            if name in env:
                return env[name]
            if name in self.consts:
                return self.consts[name]
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            raise QasmImportError(f"unknown identifier {name!r} in expression")
        if isinstance(node, ast.UnaryExpression):
            v = self.eval_expr(node.expression, env)
            op = node.op.name
            if op == "-":
                return -v
            if op == "+":
                return v
            raise QasmImportError(f"unsupported unary operator {op!r}")
        if isinstance(node, ast.BinaryExpression):
            a = self.eval_expr(node.lhs, env)
            b = self.eval_expr(node.rhs, env)
            op = node.op.name
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                if b == 0:
                    raise QasmImportError("division by zero in gate parameter")
                return a / b
            if op == "**":
                return a**b
            raise QasmImportError(f"unsupported binary operator {op!r}")
        raise QasmImportError(
            f"unsupported expression of type {type(node).__name__} "
            "(only literals, pi/tau/euler, declared constants and "
            "+ - * / ** arithmetic are supported in gate parameters)"
        )

    # -- qubit operands -----------------------------------------------------

    def resolve_operand(
        self, node: Any, qubit_env: dict[str, int] | None
    ) -> list[int]:
        """Resolve a gate qubit operand to concrete indices (len>1 = register)."""
        if isinstance(node, ast.Identifier):
            name = node.name
            if qubit_env is not None:
                if name not in qubit_env:
                    raise QasmImportError(
                        f"unknown qubit argument {name!r} inside gate definition"
                    )
                return [qubit_env[name]]
            if name not in self.qubits:
                raise QasmImportError(f"unknown qubit register {name!r}")
            off, size = self.qubits[name]
            return list(range(off, off + size))
        if isinstance(node, ast.IndexedIdentifier):
            if qubit_env is not None:
                raise QasmImportError(
                    "indexing qubit arguments inside a gate definition is "
                    "not valid OpenQASM 3"
                )
            name = node.name.name
            if name not in self.qubits:
                raise QasmImportError(f"unknown qubit register {name!r}")
            off, size = self.qubits[name]
            if len(node.indices) != 1 or len(node.indices[0]) != 1:
                raise QasmImportError(
                    f"unsupported multi-dimensional index on {name!r}"
                )
            idx_node = node.indices[0][0]
            if not isinstance(idx_node, ast.IntegerLiteral):
                idx = self.eval_expr(idx_node, {})
                if idx != int(idx):
                    raise QasmImportError(f"non-integer qubit index on {name!r}")
                idx = int(idx)
            else:
                idx = idx_node.value
            if not (0 <= idx < size):
                raise QasmImportError(
                    f"qubit index {name}[{idx}] out of range (size {size})"
                )
            return [off + idx]
        raise QasmImportError(
            f"unsupported qubit operand of type {type(node).__name__}"
        )

    # -- statements ---------------------------------------------------------

    def handle_gate_call(
        self,
        node: ast.QuantumGate,
        param_env: dict[str, float],
        qubit_env: dict[str, int] | None,
        depth: int,
    ) -> None:
        if node.modifiers:
            mods = ", ".join(m.modifier.name for m in node.modifiers)
            raise QasmImportError(
                f"gate modifiers ({mods}) are not supported; "
                "expand them before import"
            )
        raw_name = node.name.name
        name = _NAME_ALIASES.get(raw_name, raw_name)
        params = tuple(self.eval_expr(a, param_env) for a in node.arguments)
        operand_lists = [self.resolve_operand(q, qubit_env) for q in node.qubits]

        # broadcast: registers of equal size expand elementwise
        sizes = {len(ops) for ops in operand_lists if len(ops) > 1}
        if len(sizes) > 1:
            raise QasmImportError(
                f"gate {raw_name!r} broadcasts over registers of unequal sizes"
            )
        width = sizes.pop() if sizes else 1
        for w in range(width):
            qubits = tuple(
                ops[w] if len(ops) > 1 else ops[0] for ops in operand_lists
            )
            self.emit(name, raw_name, params, qubits, param_env, depth)

    def emit(
        self,
        name: str,
        raw_name: str,
        params: tuple[float, ...],
        qubits: tuple[int, ...],
        param_env: dict[str, float],
        depth: int,
    ) -> None:
        if name in self.defs and name not in REGISTRY:
            if depth >= _MAX_INLINE_DEPTH:
                raise QasmImportError(
                    f"gate definition {raw_name!r} nests deeper than "
                    f"{_MAX_INLINE_DEPTH} levels (recursive definition?)"
                )
            gdef = self.defs[name]
            if len(params) != len(gdef.params):
                raise QasmImportError(
                    f"gate {raw_name!r} takes {len(gdef.params)} parameter(s), "
                    f"got {len(params)}"
                )
            if len(qubits) != len(gdef.qubit_args):
                raise QasmImportError(
                    f"gate {raw_name!r} takes {len(gdef.qubit_args)} qubit(s), "
                    f"got {len(qubits)}"
                )
            env = dict(zip(gdef.params, params))
            qenv = dict(zip(gdef.qubit_args, qubits))
            for stmt in gdef.body:
                if isinstance(stmt, ast.QuantumGate):
                    self.handle_gate_call(stmt, env, qenv, depth + 1)
                elif isinstance(stmt, ast.QuantumPhase):
                    continue  # global phase: no effect on any observable
                elif isinstance(stmt, ast.QuantumBarrier):
                    continue
                else:
                    raise QasmImportError(
                        f"unsupported statement {type(stmt).__name__} inside "
                        f"gate definition {raw_name!r}"
                    )
            return
        if name not in REGISTRY:
            raise QasmImportError(
                f"unknown gate {raw_name!r} (not in stdgates, not defined in "
                "the file)"
            )
        self.gates.append(Gate(name, qubits, params))

    def run(self, program: ast.Program) -> Circuit:
        for stmt in program.statements:
            if isinstance(stmt, ast.Include):
                if stmt.filename not in ("stdgates.inc", "qelib1.inc"):
                    raise QasmImportError(
                        f'include "{stmt.filename}" is not supported '
                        '(only stdgates.inc)'
                    )
            elif isinstance(stmt, ast.QubitDeclaration):
                name = stmt.qubit.name
                if name in self.qubits:
                    raise QasmImportError(f"qubit register {name!r} redeclared")
                if stmt.size is None:
                    size = 1
                elif isinstance(stmt.size, ast.IntegerLiteral):
                    size = stmt.size.value
                else:
                    sz = self.eval_expr(stmt.size, {})
                    if sz != int(sz) or sz < 0:
                        raise QasmImportError(
                            f"invalid size for qubit register {name!r}"
                        )
                    size = int(sz)
                self.qubits[name] = (self.n_qubits, size)
                self.n_qubits += size
            elif isinstance(stmt, ast.QuantumGateDefinition):
                gname = stmt.name.name
                self.defs[gname] = _GateDef(
                    params=[a.name for a in stmt.arguments],
                    qubit_args=[q.name for q in stmt.qubits],
                    body=list(stmt.body),
                )
            elif isinstance(stmt, ast.QuantumGate):
                self.handle_gate_call(stmt, {}, None, 0)
            elif isinstance(stmt, ast.ConstantDeclaration):
                self.consts[stmt.identifier.name] = self.eval_expr(
                    stmt.init_expression, {}
                )
            elif isinstance(stmt, ast.QuantumMeasurementStatement):
                # recorded (reduction targets what is measured); gates are
                # unaffected — observables are supplied separately
                self.measured.extend(
                    self.resolve_operand(stmt.measure.qubit, None)
                )
            elif isinstance(
                stmt,
                (ast.QuantumBarrier, ast.QuantumPhase, ast.IODeclaration),
            ):
                continue  # no effect on observable-based reduction
            elif isinstance(stmt, ast.ClassicalDeclaration):
                # classical bits; record `bit c = measure q;` initialisers
                if isinstance(stmt.init_expression, ast.QuantumMeasurement):
                    self.measured.extend(
                        self.resolve_operand(stmt.init_expression.qubit, None)
                    )
            elif isinstance(stmt, ast.ClassicalAssignment):
                continue  # e.g. c = c1 (measure-assignments parse as
                #           QuantumMeasurementStatement, handled above)
            elif isinstance(stmt, ast.BranchingStatement):
                raise QasmImportError(
                    f"This circuit uses classical control flow"
                    f"{self._where(stmt)}.\n\n"
                    "ACTIVE reduces unitary circuits: its certified bounds "
                    "are stated for a fixed sequence of gates, and a "
                    "classically-controlled gate does not have one. Remove "
                    "the conditional, or reduce the unitary part of the "
                    "circuit separately." + self._dialect_note()
                )
            elif isinstance(stmt, ast.QuantumReset):
                raise QasmImportError(
                    f"This circuit uses reset{self._where(stmt)}.\n\n"
                    "ACTIVE reduces unitary circuits: reset discards a "
                    "qubit's state, which is not a unitary operation, so no "
                    "certified bound covers it. Remove the reset, or reduce "
                    "the unitary segments of the circuit separately."
                    + self._dialect_note()
                )
            else:
                raise QasmImportError(
                    f"unsupported OpenQASM statement: {type(stmt).__name__}"
                    f"{self._where(stmt)} "
                    "(supported: qubit/gate declarations, gate calls, "
                    "measure, barrier; not supported: control flow, "
                    "subroutines, timing)" + self._dialect_note()
                )
        circ = Circuit(self.n_qubits)
        for g in self.gates:
            circ.append(g)
        return circ


# --- OpenQASM 2.0 support -------------------------------------------------

_QASM2_HEADER = re.compile(r"OPENQASM\s+2(?:\.\d+)?\s*;")
# the first OPENQASM directive decides the dialect in "auto" mode
_ANY_HEADER = re.compile(r"^\s*OPENQASM\s+(\d+)", flags=re.MULTILINE)

# qelib1.inc gates absent from both stdgates.inc and ACTIVE's registry.
# Bodies are the CANONICAL qelib1.inc decompositions (Qiskit's copy of the
# IBM QE standard header), translated verbatim to one-line OpenQASM 3 gate
# declarations; the test suite verifies each against Qiskit's own gate
# objects exactly, including global phase. They are injected on the
# include line during conversion (keeping line numbers 1:1 with the
# source) and expanded by the ordinary importer inliner — no registry
# entries, so nothing to get wrong twice.
_QELIB1_EXTRA_DEFS: dict[str, str] = {
    "u0": "gate u0(gamma) q { U(0, 0, 0) q; }",
    "rccx": (
        "gate rccx a, b, c { u2(0, pi) c; u1(pi/4) c; cx b, c; "
        "u1(-pi/4) c; cx a, c; u1(pi/4) c; cx b, c; u1(-pi/4) c; "
        "u2(0, pi) c; }"
    ),
    "rc3x": (
        "gate rc3x a, b, c, d { u2(0, pi) d; u1(pi/4) d; cx c, d; "
        "u1(-pi/4) d; u2(0, pi) d; cx a, d; u1(pi/4) d; cx b, d; "
        "u1(-pi/4) d; cx a, d; u1(pi/4) d; cx b, d; u1(-pi/4) d; "
        "u2(0, pi) d; u1(pi/4) d; cx c, d; u1(-pi/4) d; u2(0, pi) d; }"
    ),
    "c3x": (
        "gate c3x a, b, c, d { h d; p(pi/8) a; p(pi/8) b; p(pi/8) c; "
        "p(pi/8) d; cx a, b; p(-pi/8) b; cx a, b; cx b, c; p(-pi/8) c; "
        "cx a, c; p(pi/8) c; cx b, c; p(-pi/8) c; cx a, c; cx c, d; "
        "p(-pi/8) d; cx b, d; p(pi/8) d; cx c, d; p(-pi/8) d; cx a, d; "
        "p(pi/8) d; cx c, d; p(-pi/8) d; cx b, d; p(pi/8) d; cx c, d; "
        "p(-pi/8) d; cx a, d; h d; }"
    ),
    "c3sqrtx": (
        "gate c3sqrtx a, b, c, d { h d; cu1(pi/8) a, d; h d; cx a, b; "
        "h d; cu1(-pi/8) b, d; h d; cx a, b; h d; cu1(pi/8) b, d; h d; "
        "cx b, c; h d; cu1(-pi/8) c, d; h d; cx a, c; h d; "
        "cu1(pi/8) c, d; h d; cx b, c; h d; cu1(-pi/8) c, d; h d; "
        "cx a, c; h d; cu1(pi/8) c, d; h d; }"
    ),
    "c4x": (
        "gate c4x a, b, c, d, e { h e; cu1(pi/2) d, e; h e; "
        "c3x a, b, c, d; h e; cu1(-pi/2) d, e; h e; c3x a, b, c, d; "
        "c3sqrtx a, b, c, e; }"
    ),
}
# composite bodies that reference other injected gates
_QELIB1_EXTRA_DEPS: dict[str, tuple[str, ...]] = {
    "c4x": ("c3x", "c3sqrtx"),
}


def _source_line(text: str, line_no: int | None) -> str:
    """`line N: \\`...\\`` context for an error, best-effort."""
    if not line_no:
        return ""
    lines = text.splitlines()
    if not (1 <= line_no <= len(lines)):
        return ""
    return f"line {line_no}: `{lines[line_no - 1].strip()}`"


def convert_qasm2_to_qasm3(text: str) -> str:
    """OpenQASM 2.0 -> 3.0 source transform (supported API).

    Only four surface forms differ for the supported subset: the header,
    the ``qelib1.inc`` include, ``qreg``/``creg`` declarations and the
    ``measure a -> b;`` arrow syntax. Everything else (gate calls, ``pi``
    arithmetic, ``gate`` definitions, ``barrier``) is identical in both
    languages and passes through UNTOUCHED to the one shared parser, so a
    gate can never be silently rewritten. qelib1 gates missing from
    stdgates (u0, rccx, rc3x, c3x, c3sqrtx, c4x) are provided by injecting
    their canonical qelib1 definitions as inline ``gate`` declarations on
    the include line. The transform never adds or removes a line, so
    every error line number refers to the original file.

    Constructs the reducer cannot honour are rejected here (``opaque``)
    or by the importer (``reset``, ``if(...)`` conditionals) with an
    error naming the construct and the source line.
    """
    m = re.search(r"^\s*opaque\b.*$", text, flags=re.MULTILINE)
    if m:
        line_no = text[: m.start()].count("\n") + 1
        raise QasmImportError(
            f"This circuit declares an opaque gate "
            f"({_source_line(text, line_no)}).\n\n"
            "An opaque gate has no defined unitary, so ACTIVE cannot "
            "analyse its effect on the measured observables. Replace it "
            "with a gate definition that has a body, or remove it.\n"
            "(The file was read as OpenQASM 2.0.)"
        )
    out = _QASM2_HEADER.sub("OPENQASM 3.0;", text, count=1)
    # inject only the extra qelib1 gates the file mentions (plus their
    # dependencies), all on the include line: line numbers stay 1:1
    needed: list[str] = []
    for name in _QELIB1_EXTRA_DEFS:
        if re.search(rf"\b{name}\b", out):
            for dep in _QELIB1_EXTRA_DEPS.get(name, ()):
                if dep not in needed:
                    needed.append(dep)
            if name not in needed:
                needed.append(name)
    inject = "".join(" " + _QELIB1_EXTRA_DEFS[n] for n in needed)
    seen_include = False

    def _repl_include(_m: "re.Match[str]") -> str:
        nonlocal seen_include
        if seen_include:
            return 'include "stdgates.inc";'
        seen_include = True
        return 'include "stdgates.inc";' + inject

    out = re.sub(r'include\s+"qelib1\.inc"\s*;', _repl_include, out)
    out = re.sub(
        r"\bqreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;",
        r"qubit[\2] \1;",
        out,
    )
    out = re.sub(
        r"\bcreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*;",
        r"bit[\2] \1;",
        out,
    )
    out = re.sub(
        r"\bmeasure\s+([^;>]+?)\s*->\s*([^;]+?)\s*;",
        r"\2 = measure \1;",
        out,
    )
    return out


_DIALECTS = ("auto", "qasm2", "qasm3")


def _parse(text: str, dialect: str = "auto") -> tuple[Circuit, "_Importer"]:
    if dialect not in _DIALECTS:
        raise QasmImportError(
            f"dialect must be one of {', '.join(_DIALECTS)}, got {dialect!r}"
        )
    if dialect == "auto":
        m = _ANY_HEADER.search(text)
        read_as_qasm2 = bool(m and m.group(1) == "2")
    else:
        read_as_qasm2 = dialect == "qasm2"
    original = text
    if read_as_qasm2:
        text = convert_qasm2_to_qasm3(text)
    try:
        program = openqasm3.parse(text)
    except Exception as exc:  # the antlr parser raises its own exception types
        if read_as_qasm2:
            raise QasmImportError(
                f"OpenQASM parse error: {exc}\n"
                "(The file was read as OpenQASM 2.0 and converted before "
                "parsing; the line numbers above refer to the original "
                "file — the conversion never adds or removes a line.)"
            ) from exc
        raise QasmImportError(f"OpenQASM parse error: {exc}") from exc
    imp = _Importer(source=original, read_as_qasm2=read_as_qasm2)
    circ = imp.run(program)
    # record the measured qubits on the circuit (informational only: the
    # CLI infers a default observable from it when the caller gives none)
    circ.measured_qubits = tuple(_dedup(imp.measured))
    return circ, imp


def _dedup(seq: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for q in seq:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def loads_qasm(text: str, dialect: str = "auto") -> Circuit:
    """Parse an OpenQASM 3 or 2.0 program string into a Circuit.

    The dialect is detected from the ``OPENQASM`` header (``auto``,
    default); a 2.0 file produces the identical circuit its 3.0 equivalent
    would. Pass ``dialect="qasm2"`` or ``"qasm3"`` to force the reading
    for files with a missing or wrong header.
    """
    return _parse(text, dialect)[0]


def load_qasm(path: str, dialect: str = "auto") -> Circuit:
    """Read and parse an OpenQASM 3 or 2.0 file (dialect auto-detected)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise QasmImportError(f"cannot read {path!r}: {exc}") from exc
    return loads_qasm(text, dialect)


def measured_qubits(text: str, dialect: str = "auto") -> list[int]:
    """The qubits a program measures, in program order (duplicates dropped).

    Works for OpenQASM 3 and 2.0 sources alike; register-wide measurements
    (``measure q -> c;``) expand to every qubit of the register. Useful for
    building observables that match exactly what the file measures.
    """
    _, imp = _parse(text, dialect)
    return _dedup(imp.measured)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

# Gates missing from stdgates.inc, with decompositions (verified by tests).
_EXTRA_DEFS: dict[str, str] = {
    "rzz": "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }",
    "rxx": (
        "gate rxx(theta) a, b "
        "{ h a; h b; cx a, b; rz(theta) b; cx a, b; h a; h b; }"
    ),
    "ryy": (
        "gate ryy(theta) a, b "
        "{ sdg a; h a; sdg b; h b; cx a, b; rz(theta) b; cx a, b; "
        "h a; s a; h b; s b; }"
    ),
    "iswap": "gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }",
    "sxdg": "gate sxdg a { h a; sdg a; h a; }",
    "csx": "gate csx a, b { h b; cp(pi/2) a, b; h b; }",
    "ccz": "gate ccz a, b, c { h c; ccx a, b, c; h c; }",
}

_STDGATES = {
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "cp", "cu1", "crx", "cry", "crz", "cu3", "cu",
    "swap", "ccx", "cswap",
}


def dumps_qasm(circuit: Circuit) -> str:
    """Serialize a Circuit as an OpenQASM 3.0 program string."""
    needed_defs: list[str] = []
    for g in circuit.gates:
        if g.matrix is not None:
            raise QasmExportError(
                f"gate {g.name!r} at qubits {g.qubits} is defined by a raw "
                "matrix and has no OpenQASM spelling; decompose it first"
            )
        if g.name in _STDGATES:
            continue
        if g.name in _EXTRA_DEFS:
            if _EXTRA_DEFS[g.name] not in needed_defs:
                needed_defs.append(_EXTRA_DEFS[g.name])
        else:
            raise QasmExportError(f"gate {g.name!r} cannot be exported")

    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";']
    lines.extend(needed_defs)
    if circuit.n_qubits:
        lines.append(f"qubit[{circuit.n_qubits}] q;")
    for g in circuit.gates:
        params = (
            "(" + ", ".join(repr(p) for p in g.params) + ")" if g.params else ""
        )
        targets = ", ".join(f"q[{q}]" for q in g.qubits)
        # u1/u2/u3/cu1/cu3 are "legacy" names still present in stdgates
        name = "U" if g.name == "u" else g.name
        lines.append(f"{name}{params} {targets};")
    return "\n".join(lines) + "\n"


def dump_qasm(circuit: Circuit, path: str) -> None:
    """Write a Circuit to an OpenQASM 3.0 file."""
    text = dumps_qasm(circuit)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except OSError as exc:
        raise QasmExportError(f"cannot write {path!r}: {exc}") from exc
