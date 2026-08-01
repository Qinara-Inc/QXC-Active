"""OpenQASM 3 import/export: roundtrips, subset errors, decomposition truth."""

import numpy as np
import pytest

from qxc_active import Circuit, QasmExportError, QasmImportError
from qxc_active.qasm import _EXTRA_DEFS, dumps_qasm, loads_qasm
from qxc_active import sim
from qxc_active.gates import build_unitary


GHZ = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
barrier q;
c = measure q;
"""


def _states_equal_up_to_phase(a, b, tol=1e-9):
    inner = np.vdot(a, b)
    return abs(abs(inner) - 1.0) < tol


class TestImport:
    def test_ghz(self):
        c = loads_qasm(GHZ)
        assert c.n_qubits == 3
        assert [g.name for g in c.gates] == ["h", "cx", "cx"]

    def test_parameter_expressions(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q;'
            "rz(pi/2) q[0]; rz(-3*pi/4) q[0]; rz(2**3) q[0]; rz(1.5e-3) q[0];"
        )
        assert c.gates[0].params[0] == pytest.approx(np.pi / 2)
        assert c.gates[1].params[0] == pytest.approx(-3 * np.pi / 4)
        assert c.gates[2].params[0] == pytest.approx(8.0)
        assert c.gates[3].params[0] == pytest.approx(1.5e-3)

    def test_multiple_registers(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc";'
            "qubit[2] a; qubit[3] b; cx a[1], b[0];"
        )
        assert c.n_qubits == 5
        assert c.gates[0].qubits == (1, 2)

    def test_register_broadcast(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; h q;'
        )
        assert [g.qubits for g in c.gates] == [(0,), (1,), (2,)]

    def test_two_register_broadcast(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc";'
            "qubit[2] a; qubit[2] b; cx a, b;"
        )
        assert [g.qubits for g in c.gates] == [(0, 2), (1, 3)]

    def test_custom_gate_definition_inlined(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q;'
            "gate bell a, b { h a; cx a, b; }"
            "bell q[0], q[1];"
        )
        assert [g.name for g in c.gates] == ["h", "cx"]

    def test_parameterised_gate_definition(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q;'
            "gate wiggle(theta) a { rz(theta/2) a; rx(2*theta) a; }"
            "wiggle(0.8) q[0];"
        )
        assert c.gates[0].params[0] == pytest.approx(0.4)
        assert c.gates[1].params[0] == pytest.approx(1.6)

    def test_const_declaration(self):
        c = loads_qasm(
            'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q;'
            "const float[64] my_angle = 0.25; rz(my_angle) q[0];"
        )
        assert c.gates[0].params[0] == pytest.approx(0.25)

    def test_measure_and_barrier_ignored(self):
        c = loads_qasm(GHZ)
        assert all(g.name in ("h", "cx") for g in c.gates)

    def test_reset_rejected_clearly(self):
        with pytest.raises(QasmImportError, match="not a unitary operation"):
            loads_qasm(
                'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; reset q[0];'
            )

    def test_control_flow_rejected_clearly(self):
        with pytest.raises(QasmImportError):
            loads_qasm(
                'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q; bit c;'
                "if (c == 1) { x q[0]; }"
            )

    def test_gate_modifiers_rejected_clearly(self):
        with pytest.raises(QasmImportError, match="modifier"):
            loads_qasm(
                'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q;'
                "ctrl @ x q[0], q[1];"
            )

    def test_unknown_gate_rejected_clearly(self):
        with pytest.raises(QasmImportError, match="frobnicate"):
            loads_qasm(
                'OPENQASM 3.0; include "stdgates.inc"; qubit[1] q;'
                "frobnicate q[0];"
            )

    def test_out_of_range_index_rejected(self):
        with pytest.raises(QasmImportError, match="out of range"):
            loads_qasm(
                'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[5];'
            )

    def test_parse_error_wrapped(self):
        with pytest.raises(QasmImportError):
            loads_qasm("this is not qasm at all {{{")


class TestExport:
    def test_roundtrip_preserves_semantics(self):
        c = (
            Circuit(3)
            .add("h", 0)
            .add("cx", (0, 1))
            .add("rz", 1, 0.7331)
            .add("u", 2, (0.1, -0.2, 0.3))
            .add("ccx", (0, 1, 2))
            .add("cp", (2, 0), -1.1)
        )
        c2 = loads_qasm(dumps_qasm(c))
        assert c2.n_qubits == 3
        assert _states_equal_up_to_phase(sim.statevector(c), sim.statevector(c2))

    @pytest.mark.parametrize("name", sorted(_EXTRA_DEFS))
    def test_extra_definitions_are_correct(self, name):
        """Every locally-defined export decomposition must reproduce the
        registry unitary exactly (up to global phase)."""
        from qxc_active.gates import REGISTRY

        spec = REGISTRY[name]
        params = (0.937,) * spec.n_params
        c = Circuit(spec.n_qubits)
        c.add(name, tuple(range(spec.n_qubits)), params)
        c2 = loads_qasm(dumps_qasm(c))
        # compare full unitaries column by column via basis states
        dim = 2**spec.n_qubits
        u_direct = build_unitary(name, params)
        u_decomp = np.zeros((dim, dim), dtype=complex)
        for col in range(dim):
            probe = Circuit(spec.n_qubits)
            for t in range(spec.n_qubits):
                if (col >> (spec.n_qubits - 1 - t)) & 1:
                    probe.add("x", t)
            for g in c2.gates:
                probe.append(g)
            u_decomp[:, col] = sim.statevector(probe)
        # equality up to ONE global phase for the whole matrix
        phase = np.trace(u_direct.conj().T @ u_decomp) / dim
        assert abs(abs(phase) - 1.0) < 1e-9, name
        assert np.allclose(u_decomp, phase * u_direct, atol=1e-9), name

    def test_matrix_gate_export_rejected(self):
        c = Circuit(1)
        c.add("custom", 0, matrix=np.array([[0, 1], [1, 0]], dtype=complex))
        with pytest.raises(QasmExportError, match="matrix"):
            dumps_qasm(c)

    def test_export_import_of_reduced_circuit(self):
        from qxc_active import active_reduce

        c = loads_qasm(GHZ)
        reduced, report = active_reduce(c, ["Z0*Z1"])
        text = dumps_qasm(reduced)
        assert loads_qasm(text).n_qubits == reduced.n_qubits
