"""No observable given: default to what the circuit itself measures.

The file already says what is measured; asking the user to retype it was
a usability defect. The default is Distribution(measured_qubits),
announced explicitly with the override named — never silent.
"""

from __future__ import annotations

import json

import pytest

from qxc_active import Circuit, load_qasm, loads_qasm
from qxc_active.cli import main
from qxc_active.fingerprint import circuit_fingerprint

MEASURED = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[4] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
rz(0.4) q[3];
c = measure q;
"""

SUBSET = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
bit[2] c;
h q[0];
cx q[0], q[1];
cx q[1], q[3];
c[0] = measure q[1];
c[1] = measure q[3];
"""

UNMEASURED = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""


@pytest.fixture()
def files(tmp_path):
    out = {}
    for name, src in [
        ("measured", MEASURED), ("subset", SUBSET), ("unmeasured", UNMEASURED)
    ]:
        f = tmp_path / f"{name}.qasm"
        f.write_text(src, encoding="utf-8")
        out[name] = str(f)
    return out


def test_default_fires_and_matches_explicit_run(files, capsys):
    code = main(["reduce", files["measured"], "--json"])
    err = capsys.readouterr()
    default_blob = json.loads(err.out)
    assert code in (0, 1)
    # the announcement appears exactly once (stderr in --json mode, so
    # piped JSON stays parseable)
    assert err.err.count("No observable given.") == 1

    code2 = main(
        ["reduce", files["measured"], "--sample-qubits", "0,1,2,3", "--json"]
    )
    explicit_blob = json.loads(capsys.readouterr().out)
    assert code2 == code
    for key in ("certified_bounds", "removed_gates", "output", "qubit_map"):
        assert default_blob[key] == explicit_blob[key], key


def test_announcement_text_and_placement(files, capsys):
    code = main(["reduce", files["measured"]])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert out.count("No observable given.") == 1
    assert "Using the 4 qubits this circuit measures:" in out
    assert "Distribution([0, 1, 2, 3])" in out
    assert 'Override with --observables "Z0*Z1" or --sample-qubits' in out
    # printed before the summary
    assert out.index("No observable given.") < out.index("ACTIVE reduction")


def test_unmeasured_circuit_still_errors(files, capsys):
    assert main(["reduce", files["unmeasured"]]) == 2
    err = capsys.readouterr().err
    assert "declares no measurements" in err
    assert "different observables give different reductions" in err


def test_explicit_observable_wins_no_announcement(files, capsys):
    code = main(["reduce", files["measured"], "--observables", "Z0"])
    captured = capsys.readouterr()
    assert code in (0, 1)
    assert "No observable given." not in captured.out
    assert "No observable given." not in captured.err


def test_subset_measurements_cover_exactly_that_subset(files, capsys):
    code = main(["reduce", files["subset"]])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "Using the 2 qubits this circuit measures:" in out
    assert "Distribution([1, 3])" in out


def test_measured_qubits_recorded_on_circuit():
    c = loads_qasm(MEASURED)
    assert c.measured_qubits == (0, 1, 2, 3)
    assert loads_qasm(SUBSET).measured_qubits == (1, 3)
    assert loads_qasm(UNMEASURED).measured_qubits == ()
    # copy carries it; hand-built circuits default to empty
    assert c.copy().measured_qubits == (0, 1, 2, 3)
    assert Circuit(3).measured_qubits == ()


def test_attribute_is_purely_informational():
    # equality ignores it
    a = loads_qasm(MEASURED)
    b = Circuit(4)
    for g in a.gates:
        b.append(g)
    assert a == b
    # fingerprint is byte-identical with and without measurement info,
    # so no existing report's subject hash moves
    assert circuit_fingerprint(a) == circuit_fingerprint(b)
