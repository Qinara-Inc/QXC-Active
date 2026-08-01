"""--per-observable: one maximally-reduced bundle per observable.

The load-bearing property: each observable's reduction is INDEPENDENT
and identical to running it alone — byte-identical bundles — with an
index tying them together and an honest aggregate comparison against
the joint mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qxc_active import Circuit, active_reduce
from qxc_active.batch import reduce_per_observable
from qxc_active.bundle import verify_bundle, write_bundle
from qxc_active.cli import main

# joint keeps the cx (Z1 needs it) although Z0 and Z2 alone do not:
# per-observable totals are hand-computable on this fixture
FIXTURE = """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
x q[2];
"""

OBS = ["Z0", "Z1", "Z2"]

NON_MANIFEST = (
    "reduced.qasm", "observables.json", "report.json",
    "certificate.json", "report.md",
)


@pytest.fixture()
def qasm_file(tmp_path):
    f = tmp_path / "fixture.qasm"
    f.write_text(FIXTURE, encoding="utf-8")
    return str(f)


def _circuit() -> Circuit:
    c = Circuit(3).add("h", 0).add("cx", (0, 1)).add("x", 2)
    return c


def test_bundles_byte_identical_to_separate_runs(tmp_path):
    batch_dir = tmp_path / "batch"
    reduce_per_observable(_circuit(), OBS, batch_dir)
    for i, obs in enumerate(OBS):
        solo_dir = tmp_path / f"solo{i}"
        reduced, report = active_reduce(_circuit(), [obs])
        write_bundle(solo_dir, reduced, report)
        for name in NON_MANIFEST:
            a = (batch_dir / f"obs{i}" / name).read_bytes()
            b = (solo_dir / name).read_bytes()
            assert a == b, f"obs{i}/{name} differs from a separate run"


def test_index_rows_match_input_including_failures(tmp_path):
    out = tmp_path / "o"
    index = reduce_per_observable(
        _circuit(), OBS + ["Z99"], out
    )
    rows = index["observables"]
    assert len(rows) == 4
    assert [r["error"] is None for r in rows] == [True, True, True, False]
    assert "qubit 99" in rows[3]["error"]
    # the failing row has no bundle; the others do
    assert not (out / "obs3").exists()
    assert (out / "obs0" / "reduced.qasm").is_file()
    # and the on-disk index agrees
    disk = json.loads((out / "index.json").read_text())
    assert len(disk["observables"]) == 4


def test_fully_reduced_term_in_no_execution_section(tmp_path):
    out = tmp_path / "o"
    # Z2 on a circuit with gates only on qubits 0/1: reduces to empty
    c = Circuit(3).add("h", 0).add("cx", (0, 1))
    reduce_per_observable(c, ["Z2", "Z1"], out)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "## No quantum execution required" in md
    section = md.split("## No quantum execution required")[1].split("##")[0]
    assert "`Z2`" in section
    assert "+1.000000" in section


def test_aggregate_comparison_arithmetic(tmp_path):
    # per-obs: Z0 -> 0 2q, Z1 -> 1 2q, Z2 -> 0 2q  => 1 execution
    # joint: shared circuit keeps the cx => 1 2q x 3 terms = 3
    index = reduce_per_observable(_circuit(), OBS, tmp_path / "o")
    t = index["totals"]
    assert t["two_qubit_executions"] == 1
    assert t["joint_two_qubit_gates"] == 1
    assert t["joint_two_qubit_executions"] == 3
    md = (tmp_path / "o" / "summary.md").read_text(encoding="utf-8")
    assert "1 two-qubit execution " in md
    assert "3 two-qubit executions" in md
    assert "Saving: 67%" in md


def test_jobs_parallel_identical_to_serial(tmp_path):
    a, b = tmp_path / "serial", tmp_path / "par"
    ia = reduce_per_observable(_circuit(), OBS, a, jobs=1)
    ib = reduce_per_observable(_circuit(), OBS, b, jobs=4)
    for i in range(len(OBS)):
        for name in NON_MANIFEST:
            assert (a / f"obs{i}" / name).read_bytes() == (
                b / f"obs{i}" / name
            ).read_bytes(), f"obs{i}/{name}"
    # index rows identical apart from wall time
    for ra, rb in zip(ia["observables"], ib["observables"]):
        ra, rb = dict(ra), dict(rb)
        ra.pop("wall_ms"), rb.pop("wall_ms")
        assert ra == rb


def test_verify_bundle_covers_the_tree(tmp_path):
    out = tmp_path / "o"
    reduce_per_observable(_circuit(), OBS, out)
    assert verify_bundle(out) == []
    # a byte altered inside a NESTED bundle must fail the tree check
    p = out / "obs1" / "reduced.qasm"
    p.write_bytes(p.read_bytes() + b"x q[0];\n")
    problems = verify_bundle(out)
    assert any("obs1/reduced.qasm: MISMATCH" in q for q in problems)


class TestCli:
    def test_cli_end_to_end_and_progress(self, qasm_file, tmp_path, capsys):
        out = tmp_path / "out"
        code = main(
            ["reduce", qasm_file, "--observables", *OBS,
             "--per-observable", "--out-dir", str(out)]
        )
        captured = capsys.readouterr()
        assert code == 0
        # progress on stderr, one line per observable + joint baseline
        assert captured.err.count("2q 1 ->") >= 3
        assert "[joint baseline]" in captured.err
        # summary on stdout
        assert "# Per-observable reduction" in captured.out

    def test_cli_json_suppresses_progress(self, qasm_file, tmp_path, capsys):
        out = tmp_path / "out"
        code = main(
            ["reduce", qasm_file, "--observables", *OBS,
             "--per-observable", "--out-dir", str(out), "--json"]
        )
        captured = capsys.readouterr()
        assert code == 0
        blob = json.loads(captured.out)
        assert blob["mode"] == "per-observable"
        assert captured.err == ""

    def test_exit_2_when_any_observable_rejected(self, qasm_file, tmp_path,
                                                 capsys):
        code = main(
            ["reduce", qasm_file, "--observables", "Z0", "Z99",
             "--per-observable", "--out-dir", str(tmp_path / "o")]
        )
        assert code == 2

    def test_exit_1_when_every_term_reduces_nothing(self, tmp_path, capsys):
        f = tmp_path / "stuck.qasm"
        f.write_text(
            'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\n'
            "rx(1.1) q[0];\nry(0.9) q[0];\n",
            encoding="utf-8",
        )
        code = main(
            ["reduce", str(f), "--observables", "Z0", "X0",
             "--per-observable", "--out-dir", str(tmp_path / "o")]
        )
        assert code == 1

    def test_requires_out_dir(self, qasm_file, capsys):
        code = main(
            ["reduce", qasm_file, "--observables", "Z0", "--per-observable"]
        )
        assert code == 2
        assert "--out-dir" in capsys.readouterr().err

    def test_default_joint_mode_unchanged(self, qasm_file, capsys):
        # no --per-observable: exactly the joint behaviour — one shared
        # circuit where every gate is needed by SOME observable, so
        # nothing is removed (exit 1) even though two of the three
        # observables would reduce it alone
        code = main(["reduce", qasm_file, "--observables", *OBS, "--json"])
        blob = json.loads(capsys.readouterr().out)
        assert code == 1
        assert blob["output"]["two_qubit_gates"] == 1  # cx kept for Z1
        assert blob["output"]["gates"] == 3
