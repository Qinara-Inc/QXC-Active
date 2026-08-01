"""Native OpenQASM 2.0 input: identical circuits, identical reductions.

The key property: loading a 2.0 file produces EXACTLY the same internal
circuit — and therefore exactly the same reduction result — as loading its
3.0 equivalent. Nothing in the reduction algorithm changed; this is an
input path only.
"""

import time
from pathlib import Path

import pytest

from qxc_active import active_reduce
from qxc_active.errors import QasmImportError
from qxc_active.qasm import loads_qasm, load_qasm, measured_qubits

DATA = Path(__file__).parent / "data"

BELL_V2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

BELL_V3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""

# the same mixed-gate circuit written in both languages
MIXED_V2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg a[3];
qreg b[2];
creg c[5];
u3(0.3*pi,-0.5*pi,2.0*pi) a[0];
u2(0.25,pi/4) a[1];
u1(0.7) a[2];
h b[0];
cx a[0],b[1];
cz a[1],a[2];
rx(1.5) b[0];
ry(-0.25) b[1];
rz(pi/8) a[0];
s a[1];
sdg a[2];
t b[0];
tdg b[1];
swap a[0],a[2];
ccx a[0],a[1],b[0];
crz(0.6) a[2],b[1];
cu1(0.4) a[0],a[1];
barrier a;
measure a -> c[0];
"""

MIXED_V3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] a;
qubit[2] b;
bit[5] c;
u3(0.3*pi,-0.5*pi,2.0*pi) a[0];
u2(0.25,pi/4) a[1];
u1(0.7) a[2];
h b[0];
cx a[0], b[1];
cz a[1], a[2];
rx(1.5) b[0];
ry(-0.25) b[1];
rz(pi/8) a[0];
s a[1];
sdg a[2];
t b[0];
tdg b[1];
swap a[0], a[2];
ccx a[0], a[1], b[0];
crz(0.6) a[2], b[1];
cu1(0.4) a[0], a[1];
barrier a;
c[0] = measure a;
"""


class TestBasicParsing:
    def test_bell_loads(self):
        c = loads_qasm(BELL_V2)
        assert c.n_qubits == 2
        assert [g.name for g in c.gates] == ["h", "cx"]

    def test_measured_qubits_read_exactly(self):
        assert measured_qubits(BELL_V2) == [0, 1]

    def test_register_measure_broadcast(self):
        src = (
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4];\n'
            "h q[0];\nmeasure q -> c;\n"
        )
        assert measured_qubits(src) == [0, 1, 2, 3]

    def test_measure_order_preserved(self):
        src = (
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; creg c[3];\n'
            "measure q[2] -> c[2];\nmeasure q[0] -> c[0];\n"
        )
        assert measured_qubits(src) == [2, 0]

    def test_qelib1_gate_names(self):
        src = (
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[3];\n'
            "id q[0]; x q[0]; y q[0]; z q[0]; h q[0]; s q[0]; sdg q[0];\n"
            "t q[0]; tdg q[0]; sx q[0];\n"
            "rx(0.1) q[0]; ry(0.2) q[1]; rz(0.3) q[2];\n"
            "u1(0.4) q[0]; u2(0.5,0.6) q[1]; u3(0.7,0.8,0.9) q[2];\n"
            "p(0.15) q[0]; U(0.1,0.2,0.3) q[1]; CX q[0],q[1];\n"
            "cx q[0],q[1]; cy q[0],q[1]; cz q[0],q[1]; ch q[0],q[1];\n"
            "swap q[0],q[1]; crx(0.1) q[0],q[1]; cry(0.2) q[0],q[1];\n"
            "crz(0.3) q[0],q[1]; cu1(0.4) q[0],q[1];\n"
            "cu3(0.1,0.2,0.3) q[0],q[1]; cp(0.5) q[0],q[1];\n"
            "rxx(0.6) q[0],q[1]; rzz(0.7) q[0],q[1];\n"
            "ccx q[0],q[1],q[2]; cswap q[0],q[1],q[2];\n"
        )
        c = loads_qasm(src)
        assert c.n_gates == 34
        for g in c.gates:
            g.unitary()  # every gate resolves to a concrete unitary

    def test_gate_definition_inlined(self):
        src = (
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[2];\n'
            "gate bell a, b { h a; cx a, b; }\n"
            "bell q[0], q[1];\n"
        )
        assert [g.name for g in loads_qasm(src).gates] == ["h", "cx"]

    def test_version_2_without_minor(self):
        src = 'OPENQASM 2; include "qelib1.inc"; qreg q[1]; h q[0];'
        assert loads_qasm(src).n_gates == 1


class TestEquivalence:
    """THE key property: 2.0 and 3.0 sources give identical results."""

    @pytest.mark.parametrize(
        "v2,v3", [(BELL_V2, BELL_V3), (MIXED_V2, MIXED_V3)]
    )
    def test_identical_internal_circuit(self, v2, v3):
        c2, c3 = loads_qasm(v2), loads_qasm(v3)
        assert c2 == c3  # exact: names, qubits, parameters

    @pytest.mark.parametrize(
        "v2,v3,obs",
        [
            (BELL_V2, BELL_V3, ["Z0*Z1"]),
            (MIXED_V2, MIXED_V3, ["Z0", "X2*Z4"]),
        ],
    )
    def test_identical_reduction_result(self, v2, v3, obs):
        r2, rep2 = active_reduce(loads_qasm(v2), obs)
        r3, rep3 = active_reduce(loads_qasm(v3), obs)
        assert r2 == r3
        d2, d3 = rep2.to_dict(), rep3.to_dict()
        d2.pop("timings_ms"), d3.pop("timings_ms")
        assert d2 == d3

    def test_identical_reduction_approx_mode(self):
        r2, rep2 = active_reduce(loads_qasm(MIXED_V2), ["Z0"], epsilon=0.01)
        r3, rep3 = active_reduce(loads_qasm(MIXED_V3), ["Z0"], epsilon=0.01)
        assert r2 == r3
        assert rep2.certified_bounds == rep3.certified_bounds

    def test_measured_qubits_agree(self):
        assert measured_qubits(MIXED_V2) == measured_qubits(MIXED_V3)


class TestUnsupportedConstructsFailClearly:
    """Rejections must be actionable: construct, source line, and why it
    is out of scope — matched on message content, not just type."""

    def test_conditional_names_line_and_scope(self):
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n'
            "measure q[0] -> c[0];\nif (c == 1) x q[0];\n"
        )
        with pytest.raises(QasmImportError) as ei:
            loads_qasm(src)
        msg = str(ei.value)
        assert "classical control flow" in msg
        assert "line 6: `if (c == 1) x q[0];`" in msg
        assert "certified bounds" in msg
        assert "read as OpenQASM 2.0" in msg

    def test_reset_names_line_and_scope(self):
        src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nreset q[0];\n'
        with pytest.raises(QasmImportError) as ei:
            loads_qasm(src)
        msg = str(ei.value)
        assert "reset" in msg
        assert "line 4: `reset q[0];`" in msg
        assert "not a unitary operation" in msg
        assert "read as OpenQASM 2.0" in msg

    def test_opaque_names_line_and_scope(self):
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
            "opaque mystery(a) q;\n"
        )
        with pytest.raises(QasmImportError) as ei:
            loads_qasm(src)
        msg = str(ei.value)
        assert "opaque gate" in msg
        assert "line 4: `opaque mystery(a) q;`" in msg
        assert "no defined unitary" in msg
        assert "read as OpenQASM 2.0" in msg

    def test_qasm3_reset_message_has_no_dialect_note(self):
        src = (
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\n'
            "reset q[0];\n"
        )
        with pytest.raises(QasmImportError) as ei:
            loads_qasm(src)
        msg = str(ei.value)
        assert "not a unitary operation" in msg
        assert "read as OpenQASM 2.0" not in msg

    def test_unknown_gate_named(self):
        src = (
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[3]; '
            "mystery_gate q[0],q[1];"
        )
        with pytest.raises(QasmImportError, match="mystery_gate"):
            loads_qasm(src)

    def test_parse_error_names_qasm2_reading(self):
        # header says 2.0 and the converted source still fails to parse:
        # the error must say the file was read as OpenQASM 2
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'
            "h q[0\n"  # syntax error
        )
        with pytest.raises(QasmImportError) as ei:
            loads_qasm(src)
        msg = str(ei.value)
        assert "parse error" in msg
        assert "read as OpenQASM 2.0" in msg
        assert "line numbers above refer to the original file" in msg

    def test_measure_and_barrier_still_silently_dropped(self):
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\nbarrier q;\nmeasure q -> c;\n"
        )
        c = loads_qasm(src)
        assert [g.name for g in c.gates] == ["h"]


class TestDialectOverride:
    RCCX_WITH_HEADER = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
        "rccx q[0],q[1],q[2];\n"
    )
    HEADERLESS = 'include "qelib1.inc";\nqreg q[3];\nrccx q[0],q[1],q[2];\n'

    def test_auto_detects_qasm2_header(self):
        c = loads_qasm(self.RCCX_WITH_HEADER)
        assert c.n_gates == 9  # canonical rccx body

    def test_auto_detects_qasm3_header(self):
        c = loads_qasm(BELL_V3)
        assert [g.name for g in c.gates] == ["h", "cx"]

    def test_force_qasm2_on_headerless_file(self):
        # without the header auto cannot know the qelib1 extras are in
        # play; the explicit override supplies them
        c = loads_qasm(self.HEADERLESS, dialect="qasm2")
        assert c.n_gates == 9

    def test_headerless_auto_lacks_qelib1_extras(self):
        with pytest.raises(QasmImportError, match="rccx"):
            loads_qasm(self.HEADERLESS)

    def test_force_qasm3_on_qasm2_header(self):
        # forcing qasm3 skips the conversion: the qelib1-only gate is
        # then unknown, exactly as a real QASM 3 reader would see it
        with pytest.raises(QasmImportError, match="rccx"):
            loads_qasm(self.RCCX_WITH_HEADER, dialect="qasm3")

    def test_force_qasm3_on_common_subset_still_works(self):
        # qreg/creg/measure-arrow are legacy-valid OpenQASM 3, so a plain
        # QASM 2 file read as QASM 3 still yields the identical circuit
        c = loads_qasm(BELL_V2, dialect="qasm3")
        assert [g.name for g in c.gates] == ["h", "cx"]

    def test_invalid_dialect_rejected(self):
        with pytest.raises(QasmImportError, match="dialect"):
            loads_qasm(BELL_V2, dialect="qasm4")

    def test_load_qasm_accepts_dialect(self, tmp_path):
        f = tmp_path / "headerless.qasm"
        f.write_text(self.HEADERLESS, encoding="utf-8")
        assert load_qasm(str(f), dialect="qasm2").n_gates == 9

    def test_measured_qubits_accepts_dialect(self):
        src = (
            'include "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\nmeasure q -> c;\n"
        )
        assert measured_qubits(src, dialect="qasm2") == [0, 1]


class TestQelib1ExtraGates:
    """u0 / rccx / rc3x / c3x / c3sqrtx / c4x: qelib1 gates missing from
    stdgates, provided by injecting their canonical qelib1 bodies. Parse
    behaviour here; the unitaries are cross-checked against Qiskit in
    test_crosscheck_qiskit.py (exactly, including global phase)."""

    def test_all_extras_parse_and_expand(self):
        c = load_qasm(str(DATA / "qasm2" / "qelib1_extras.qasm"))
        assert c.n_qubits == 5
        # every emitted gate is a registry gate with a concrete unitary
        for g in c.gates:
            g.unitary()

    def test_u0_is_identity(self):
        src = (
            'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; u0(3.7) q[0];'
        )
        c = loads_qasm(src)
        assert len(c.gates) == 1
        import numpy as np

        assert np.allclose(c.gates[0].unitary(), np.eye(2))

    def test_user_definition_of_same_name_wins(self):
        # a file may define its own rccx: the file's definition applies to
        # calls after it, exactly as in plain QASM
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
            "gate rccx a,b,c { h c; }\n"
            "rccx q[0],q[1],q[2];\n"
        )
        c = loads_qasm(src)
        assert [g.name for g in c.gates] == ["h"]

    def test_line_numbers_survive_injection(self):
        # the injected definitions share the include line, so an error
        # after them still reports the ORIGINAL line number
        src = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'
            "rccx q[0],q[1],q[2];\nreset q[0];\n"
        )
        with pytest.raises(QasmImportError, match=r"line 5: `reset q\[0\];`"):
            loads_qasm(src)


FIXTURES = {
    "bell.qasm": {"qubits": 2, "names": ["h", "cx"], "measured": [0, 1]},
    "broadcast.qasm": {
        "qubits": 4,
        "names": ["h", "h", "h", "h", "cx"],
        "measured": [0, 1, 2, 3],
    },
    "custom_gate.qasm": {
        "qubits": 3,
        "names": ["h", "cx", "cx", "cx", "ccx"],
        "measured": [0, 1, 2],
    },
    "rotations.qasm": {
        "qubits": 4,
        "names": [
            "u3", "u2", "u1", "rx", "ry", "rz", "crz", "cu1", "cu3",
            "rxx", "rzz",
        ],
        "measured": [0, 2, 3],
    },
    "qelib1_extras.qasm": {"qubits": 5, "names": None, "measured": []},
}


class TestFixtureCorpus:
    """Small committed QASM 2 corpus: tested without any network fetch."""

    @pytest.mark.parametrize("fname", sorted(FIXTURES))
    def test_fixture_parses_as_expected(self, fname):
        text = (DATA / "qasm2" / fname).read_text(encoding="utf-8")
        c = loads_qasm(text)
        want = FIXTURES[fname]
        assert c.n_qubits == want["qubits"]
        if want["names"] is not None:
            assert [g.name for g in c.gates] == want["names"]
        assert measured_qubits(text) == want["measured"]

    def test_corpus_is_complete(self):
        found = {p.name for p in (DATA / "qasm2").glob("*.qasm")}
        assert found == set(FIXTURES)


BENCHPRESS = (
    Path(__file__).parent.parent
    / "benchmarks" / "external" / "benchpress" / "qasm"
)


@pytest.mark.skipif(
    not BENCHPRESS.exists(),
    reason="Benchpress corpus not fetched (benchmarks/external/benchpress)",
)
def test_benchpress_corpus_pass_rate_does_not_regress():
    """Every failure in the corpus must be an out-of-scope construct
    (classical control flow), never a parsing regression."""
    families = [
        "feynman", "qaoa", "qft", "qv", "square-heisenberg",
        "clifford", "dtc", "bigint",
    ]
    ok = failures = 0
    for fam in families:
        for f in sorted((BENCHPRESS / fam).glob("*.qasm")):
            try:
                loads_qasm(f.read_text(encoding="utf-8"))
                ok += 1
            except QasmImportError as exc:
                failures += 1
                assert "classical control flow" in str(exc), (
                    f"{fam}/{f.name} failed for a new reason: {exc}"
                )
    assert ok + failures > 0
    # measured pass rate at promotion time; must never drop
    assert ok / (ok + failures) >= 0.98


@pytest.fixture(scope="module")
def nh3_text():
    return (DATA / "nh3_sto3g_16qubits.qasm").read_text(encoding="utf-8")


class TestRealChemistryCircuit:
    """nh3_sto3g_16qubits.qasm: 16 qubits, ~6.7k gates, all qubits measured.

    The point is that it LOADS and reduces correctly — near-zero reduction
    is expected because everything is measured.
    """

    def test_loads(self, nh3_text):
        c = loads_qasm(nh3_text)
        assert c.n_qubits == 16
        assert c.n_gates == 6676  # 3358 u3 + 3318 cx
        names = {g.name for g in c.gates}
        assert names == {"u3", "cx"}

    def test_measures_all_16_qubits(self, nh3_text):
        assert sorted(measured_qubits(nh3_text)) == list(range(16))

    def test_reduces_cleanly(self, nh3_text):
        c = loads_qasm(nh3_text)
        t0 = time.perf_counter()
        reduced, report = active_reduce(c, ["Z0"])
        dt = time.perf_counter() - t0
        assert dt < 60, f"reduction took {dt:.1f}s (test budget is 60s)"
        assert report.input_metrics["gates"] == 6676
        assert reduced.n_gates <= c.n_gates
        assert all(b < 1e-6 for b in report.certified_bounds)

    def test_load_qasm_from_path(self):
        c = load_qasm(str(DATA / "nh3_sto3g_16qubits.qasm"))
        assert c.n_qubits == 16
