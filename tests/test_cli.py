"""CLI behaviour: reduce, info, outputs, error handling, exit codes."""

import json

import pytest

from qxc_active.cli import main

GHZ = """OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
rz(pi/4) q[3];
"""


@pytest.fixture()
def qasm_file(tmp_path):
    f = tmp_path / "ghz.qasm"
    f.write_text(GHZ, encoding="utf-8")
    return str(f)


def test_reduce_summary(qasm_file, capsys):
    assert main(["reduce", qasm_file, "--observables", "Z0"]) == 0
    out = capsys.readouterr().out
    assert "ACTIVE reduction" in out and "certified" in out


BUNDLE_FILES = (
    "reduced.qasm", "observables.json", "report.json",
    "certificate.json", "report.md", "manifest.json",
)


def test_out_dir_writes_the_full_bundle(qasm_file, tmp_path, capsys):
    out = tmp_path / "bundle"
    code = main(
        ["reduce", qasm_file, "--observables", "Z0*Z1",
         "--out-dir", str(out)]
    )
    assert code == 0
    for name in BUNDLE_FILES:
        assert (out / name).is_file(), name
    assert "OPENQASM 3.0" in (out / "reduced.qasm").read_text()
    blob = json.loads((out / "report.json").read_text())
    assert blob["input"]["gates"] == 4
    assert blob["epsilon"] == 0.0
    obs = json.loads((out / "observables.json").read_text())
    assert obs["measurement"]["observables"][0]["reduced_terms"]
    cert = json.loads((out / "certificate.json").read_text())
    assert cert["certified_bounds"] == blob["certified_bounds"]
    assert len(cert["steps"]) == len(blob["removed_gates"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["files"]) == set(BUNDLE_FILES) - {"manifest.json"}


def test_observables_json_written_even_when_nothing_removed(tmp_path, capsys):
    f = tmp_path / "stuck.qasm"
    f.write_text(
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\n'
        "rx(1.1) q[0];\nry(0.9) q[0];\n",
        encoding="utf-8",
    )
    out = tmp_path / "bundle"
    code = main(
        ["reduce", str(f), "--observables", "Z0", "--out-dir", str(out)]
    )
    assert code == 1  # nothing removed — still a complete bundle
    assert (out / "observables.json").is_file()
    obs = json.loads((out / "observables.json").read_text())
    assert obs["measurement"]["observables"][0]["reduced"] == "1*Z0"


def test_bundle_is_deterministic_except_manifest(qasm_file, tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        assert main(
            ["reduce", qasm_file, "--observables", "Z0*Z1",
             "--out-dir", str(out)]
        ) == 0
    for name in BUNDLE_FILES:
        if name == "manifest.json":
            continue
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    # identical apart from run metadata (timestamp, timings)
    ma.pop("created_utc"), mb.pop("created_utc")
    ma.pop("timings_ms"), mb.pop("timings_ms")
    assert ma == mb


def test_verify_bundle_ok_and_tampered(qasm_file, tmp_path, capsys):
    out = tmp_path / "bundle"
    assert main(
        ["reduce", qasm_file, "--observables", "Z0*Z1",
         "--out-dir", str(out)]
    ) == 0
    assert main(["verify-bundle", str(out)]) == 0
    assert "bundle verified" in capsys.readouterr().out
    # alter the circuit by one gate
    p = out / "reduced.qasm"
    p.write_bytes(p.read_bytes() + b"x q[0];\n")
    assert main(["verify-bundle", str(out)]) == 2
    err = capsys.readouterr().err
    assert "reduced.qasm: MISMATCH" in err
    assert "FAILED" in err


def test_removed_output_flag_fails_loudly(qasm_file, tmp_path, capsys):
    code = main(
        ["reduce", qasm_file, "--observables", "Z0",
         "--output", str(tmp_path / "r.qasm")]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "--output was removed" in err
    assert "renumbers qubits" in err
    assert "--out-dir" in err
    assert not (tmp_path / "r.qasm").exists()


def test_removed_report_flag_fails_loudly(qasm_file, tmp_path, capsys):
    code = main(
        ["reduce", qasm_file, "--observables", "Z0",
         "--report", str(tmp_path / "rep.json")]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "--report was removed" in err
    assert "report.json" in err and "--out-dir" in err


def test_print_qasm_warns_on_stderr_when_renumbered(qasm_file, capsys):
    # Z3 on the 4-qubit GHZ file: qubits 0-2 are dropped and qubit 3 is
    # renumbered to 0 by compaction
    code = main(
        ["reduce", qasm_file, "--observables", "Z3", "--print-qasm"]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "OPENQASM 3.0" in captured.out
    assert "warning: compaction renumbered qubits" in captured.err
    assert "--out-dir" in captured.err


def test_print_qasm_no_warning_when_indices_unchanged(qasm_file, capsys):
    # Z0: the surviving qubits keep their indices, so no warning
    code = main(
        ["reduce", qasm_file, "--observables", "Z0", "--print-qasm"]
    )
    assert code == 0
    assert "warning" not in capsys.readouterr().err


def test_reduce_singular_observable_alias(qasm_file, capsys):
    # --observable (singular) is the README's first command; it must work
    # and not die as an ambiguous prefix of --observables/--observables-file
    assert main(["reduce", qasm_file, "--observable", "Z0"]) == 0
    out = capsys.readouterr().out
    assert "ACTIVE reduction" in out


def test_reduce_json_mode(qasm_file, capsys):
    assert main(["reduce", qasm_file, "--observables", "Z0", "--json"]) == 0
    blob = json.loads(capsys.readouterr().out)
    assert blob["input"]["qubits"] == 4


def test_reduce_epsilon(qasm_file, capsys):
    code = main(
        ["reduce", qasm_file, "--observables", "Z0", "--epsilon", "0.01"]
    )
    assert code == 0
    assert "epsilon=0.01" in capsys.readouterr().out


def test_observables_file(qasm_file, tmp_path, capsys):
    obs = tmp_path / "obs.txt"
    obs.write_text("Z0   # comment\n\nX0*X1\n", encoding="utf-8")
    code = main(
        ["reduce", qasm_file, "--observables-file", str(obs), "--json"]
    )
    assert code == 0
    blob = json.loads(capsys.readouterr().out)
    assert len(blob["observables"]) == 2


def test_print_qasm(qasm_file, capsys):
    code = main(
        ["reduce", qasm_file, "--observables", "Z0", "--print-qasm"]
    )
    assert code == 0
    assert 'include "stdgates.inc";' in capsys.readouterr().out


def test_no_compact(qasm_file, capsys):
    code = main(
        ["reduce", qasm_file, "--observables", "Z0", "--json", "--no-compact"]
    )
    assert code == 0
    blob = json.loads(capsys.readouterr().out)
    assert blob["output"]["qubits"] == 4


def test_info(qasm_file, capsys):
    assert main(["info", qasm_file]) == 0
    blob = json.loads(capsys.readouterr().out)
    assert blob == {"qubits": 4, "gates": 4, "depth": 3, "two_qubit_gates": 2}


def test_exit_code_1_when_nothing_removed(tmp_path, capsys):
    # canonical QAOA ring + global observable: provably nothing removable;
    # pipelines branch on the exit code
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', "qubit[4] q;"]
    for q in range(4):
        lines.append(f"h q[{q}];")
    for q in range(4):
        lines.append(f"rzz(0.7) q[{q}], q[{(q + 1) % 4}];")
    for q in range(4):
        lines.append(f"rx(0.4) q[{q}];")
    f = tmp_path / "qaoa.qasm"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    obs = " + ".join(f"Z{q}" for q in range(4))
    code = main(["reduce", str(f), "--observables", obs])
    out = capsys.readouterr().out
    assert "No reduction possible" in out
    assert code == 1


def test_explain_leads_with_conclusion(qasm_file, capsys):
    assert main(
        ["reduce", qasm_file, "--observables", "Z0", "--explain"]
    ) == 0
    out = capsys.readouterr().out
    # the explanation opens with the outcome, not the statistics
    explain_part = out.split("\n\n", 1)[1]
    assert explain_part.startswith("Removed ")


def test_missing_observables_error(qasm_file, capsys):
    # the GHZ fixture declares no measurements: still an error, but one
    # that explains what an observable is rather than listing flags
    assert main(["reduce", qasm_file]) == 2
    err = capsys.readouterr().err
    assert "declares no measurements" in err
    assert "the quantity you want the answer to" in err
    assert '--observables "Z0"' in err


def test_missing_file_error(capsys):
    assert main(["reduce", "no_such_file.qasm", "--observables", "Z0"]) == 2
    assert "error:" in capsys.readouterr().err


def test_bad_observable_error(qasm_file, capsys):
    assert main(["reduce", qasm_file, "--observables", "Q99"]) == 2
    assert "error:" in capsys.readouterr().err


def test_bad_epsilon_error(qasm_file, capsys):
    assert main(
        ["reduce", qasm_file, "--observables", "Z0", "--epsilon", "-1"]
    ) == 2


def _run_with_closing_reader(cli_args: list) -> tuple[int, bytes]:
    """Run the CLI in a subprocess, read one chunk of stdout, close the
    pipe (like `| head`), and return (exit code, stderr)."""
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "from qxc_active.cli import main\n"
        f"sys.exit(main({cli_args!r}))\n"
    )
    proc = subprocess.Popen(
        [_sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdout.read(512)  # a reader that stops after a taste
    proc.stdout.close()
    err = proc.stderr.read()
    proc.stderr.close()
    rc = proc.wait(timeout=120)
    return rc, err


def _big_reducible_qasm(tmp_path) -> str:
    # thousands of removable gates: the --json report is far larger than
    # any pipe buffer, so the broken pipe fires mid-print
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', "qubit[16] q;"]
    for i in range(3000):
        lines.append(f"rz({0.1 + 0.0001 * i}) q[{1 + i % 15}];")
        lines.append(f"cp({0.2 + 0.0001 * i}) q[{1 + i % 15}], "
                     f"q[{1 + (i + 1) % 15}];")
    f = tmp_path / "big.qasm"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(f)


def _big_stuck_qasm(tmp_path) -> str:
    # a large circuit where nothing is removable under the global
    # observable set (QAOA ring), so exit code 1 with a big report
    n, p = 20, 5
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', f"qubit[{n}] q;"]
    lines += [f"h q[{q}];" for q in range(n)]
    for layer in range(p):
        lines += [
            f"rzz({0.7 + 0.31 * layer}) q[{q}], q[{(q + 1) % n}];"
            for q in range(n)
        ]
        lines += [f"rx({0.4 + 0.17 * layer}) q[{q}];" for q in range(n)]
    f = tmp_path / "stuck.qasm"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(f)


def test_broken_pipe_no_traceback_and_exit_0(tmp_path):
    f = _big_reducible_qasm(tmp_path)
    rc, err = _run_with_closing_reader(
        ["reduce", f, "--observables", "Z0", "--json"]
    )
    assert b"Traceback" not in err, err[:500]
    assert b"Exception ignored" not in err, err[:500]
    assert b"BrokenPipeError" not in err, err[:500]
    assert rc == 0  # the semantic result survives the broken pipe


def test_broken_pipe_preserves_no_reduction_exit_code(tmp_path):
    f = _big_stuck_qasm(tmp_path)
    obs = []
    for q in range(20):
        obs += [f"Z{q}"]
    rc, err = _run_with_closing_reader(
        ["reduce", f, "--observables", *obs, "--json"]
    )
    assert b"Traceback" not in err, err[:500]
    assert b"Exception ignored" not in err, err[:500]
    assert rc == 1  # a piped no-reduction run must still report it


def test_exit_2_on_rejected_input(tmp_path, capsys):
    f = tmp_path / "reset.qasm"
    f.write_text(
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nreset q[0];\n',
        encoding="utf-8",
    )
    assert main(["reduce", str(f), "--observables", "Z0"]) == 2
    assert "not a unitary operation" in capsys.readouterr().err


def test_exit_3_on_budget_violation(qasm_file, capsys, monkeypatch):
    # should be unreachable in reality; the CLI contract for it is not
    import qxc_active.cli as cli_mod
    from qxc_active.errors import BudgetViolationError

    def _boom(*a, **kw):
        raise BudgetViolationError("charge 2.0 exceeds epsilon 0.01")

    monkeypatch.setattr(cli_mod, "active_reduce", _boom)
    code = main(
        ["reduce", qasm_file, "--observables", "Z0", "--epsilon", "0.01"]
    )
    assert code == 3
    err = capsys.readouterr().err
    assert "INTERNAL INVARIANT VIOLATED" in err


def test_help_documents_exit_codes(capsys):
    import pytest as _pytest

    with _pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "exit codes:" in out
    assert "nothing removed" in out
    assert "do not trust" in out
