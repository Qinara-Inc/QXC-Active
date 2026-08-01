"""The explain-your-work report: narrative + structured explanation block."""

import json

import numpy as np
import pytest

from conftest import random_circuit
from qxc_active import Circuit, Distribution, active_reduce


@pytest.fixture()
def ghz_case():
    c = Circuit(4).add("h", 0).add("cx", (0, 1)).add("cx", (1, 2)).add(
        "rz", 3, 0.7
    )
    return c, active_reduce(c, ["Z0*Z1"])


def test_explain_renders_key_facts(ghz_case):
    c, (reduced, report) = ghz_case
    text = report.explain()
    assert "WHAT HAPPENED" in text
    assert "WHY GATES WERE REMOVED" in text
    assert "ERROR ACCOUNTING" in text
    assert "proven upper bounds" in text
    # counts must match the structured data, properly pluralised
    for reason, count in report.removed_by_reason().items():
        noun = "gate" if count == 1 else "gates"
        assert f"{count:>5} {noun}:" in text
    assert "gate(s)" not in text and "1 gates" not in text


def test_removed_by_reason_totals(ghz_case):
    _, (reduced, report) = ghz_case
    assert sum(report.removed_by_reason().values()) == len(report.removed_gates)


def test_kept_gate_ledger_ordering():
    rng = np.random.default_rng(5)
    circ = random_circuit(rng, 4, 25)
    _, report = active_reduce(circ, ["Z0"])
    ledger = report.kept_gate_ledger()
    assert len(ledger) == len(report.influence_scores)
    # bounded entries must be sorted by descending influence
    bounded = [r["max_influence"] for r in ledger if r["max_influence"] is not None]
    assert bounded == sorted(bounded, reverse=True)


def test_explanation_block_in_json(ghz_case):
    _, (reduced, report) = ghz_case
    blob = json.loads(report.to_json())
    exp = blob["explanation"]
    assert exp["removed_by_reason"] == report.removed_by_reason()
    assert set(exp["reason_glossary"]) == set(exp["removed_by_reason"])
    assert exp["text"] == report.explain()
    assert exp["measurement_kinds"] == ["expectation"]
    assert isinstance(exp["kept_gates"], list)


def test_distribution_wording():
    c = Circuit(2).add("h", 0).add("cx", (0, 1))
    _, report = active_reduce(c, [Distribution([0, 1])])
    text = report.explain()
    assert "every outcome probability" in text
    blob = json.loads(report.to_json())
    assert blob["explanation"]["measurement_kinds"] == ["distribution"]


def test_exact_mode_wording(ghz_case):
    _, (reduced, report) = ghz_case
    assert "Exact mode" in report.explain()


def test_approx_mode_wording():
    rng = np.random.default_rng(8)
    circ = random_circuit(rng, 4, 20)
    _, report = active_reduce(circ, ["Z0"], epsilon=0.05)
    text = report.explain()
    assert "epsilon = 0.05" in text


def test_coarsening_note():
    rng = np.random.default_rng(9)
    circ = random_circuit(rng, 5, 40)
    _, report = active_reduce(circ, ["Z0*X1*Y2"], max_terms=16)
    if report.coarsening_events:
        assert "size cap" in report.explain()


def test_cli_explain_flag(tmp_path, capsys):
    from qxc_active.cli import main

    f = tmp_path / "c.qasm"
    f.write_text(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q;\n'
        "h q[0]; cx q[0], q[1]; rz(0.3) q[2];\n",
        encoding="utf-8",
    )
    assert main(["reduce", str(f), "--observables", "Z1", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "ACTIVE reduction" in out and "WHAT HAPPENED" in out


def _qaoa_ring_qasm(n: int, p: int) -> str:
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', f"qubit[{n}] q;"]
    lines += [f"h q[{q}];" for q in range(n)]
    for layer in range(p):
        lines += [
            f"rzz({0.7 + 0.31 * layer}) q[{q}], q[{(q + 1) % n}];"
            for q in range(n)
        ]
        lines += [f"rx({0.4 + 0.17 * layer}) q[{q}];" for q in range(n)]
    return "\n".join(lines) + "\n"


def test_cli_explain_prints_diagnosis_exactly_once(tmp_path, capsys):
    """Defect 1: --explain adds to summary(), never repeats it."""
    from qxc_active.cli import main

    f = tmp_path / "qaoa.qasm"
    f.write_text(_qaoa_ring_qasm(6, 2), encoding="utf-8")
    obs = " + ".join(f"Z{q}" for q in range(6))
    code = main(["reduce", str(f), "--observables", obs, "--explain"])
    assert code == 1  # no reduction
    out = capsys.readouterr().out
    # the diagnosis appears exactly once (in summary), not again in explain
    assert out.count("Why: ") == 1
    assert out.count("What to try:") == 1
    assert out.count("for this circuit and observable.") == 1
    # explain still leads with the conclusion and adds the ledger
    assert "gate-by-gate account" in out
    assert "ERROR ACCOUNTING" in out


def test_saturated_ledger_is_not_presented_as_a_ranking():
    """Defect 2: every row clamped to the physical maximum -> prose, not
    a ranking of identical ceilings.

    Alternating x/y on one qubit: every conjugation flips the sign of the
    back-propagated Pauli, so every gate's influence bound is exactly 2 —
    the trivial maximum for a norm-1 observable. clifford_resynthesis is
    off so the (all-Clifford) sequence is kept rather than resynthesised.
    """
    c = Circuit(1)
    for name in ("x", "y", "x", "y"):
        c.add(name, 0)
    _, report = active_reduce(c, ["Z0"], clifford_resynthesis=False)
    assert report.no_reduction
    ledger = report.kept_gate_ledger()
    assert all(r["max_influence"] is not None for r in ledger)
    text = report.explain()
    flat = " ".join(text.split())  # undo line wrapping for phrase checks
    assert "most load-bearing first" not in text
    assert "gate #" not in text  # no per-gate ceiling rows
    assert "could be bounded below the trivial maximum" in flat
    assert "worst-case over all input states" in flat
    assert "none was provably removable" in flat
    # the JSON ledger is untouched: still one raw entry per kept gate
    blob = json.loads(report.to_json())
    assert len(blob["explanation"]["kept_gates"]) == len(ledger)


def test_mixed_ledger_folds_saturated_rows_and_ranks_the_rest():
    """Defect 2, mixed case: many rows at the ceiling plus a genuinely
    ranked tail — the ceiling rows fold into one line, the tail ranks."""
    c = Circuit(1)
    for name, th in [("rx", 2.2), ("ry", 1.9), ("rx", 2.5), ("ry", 2.1),
                     ("rx", 1.8)]:
        c.add(name, 0, th)
    _, report = active_reduce(c, ["Z0"])
    bounded = [
        r for r in report.kept_gate_ledger()
        if r["max_influence"] is not None
    ]
    n_sat = sum(1 for r in bounded if r["max_influence"] >= 2.0)
    assert n_sat >= 2 and n_sat < len(bounded), "probe circuit drifted"
    text = report.explain()
    flat = " ".join(text.split())
    # one summary line for the ceiling rows, with the substance intact
    assert f"{n_sat} gates: no bound below the trivial maximum" in flat
    assert "worst-case bounds" in flat
    assert "provably removable" in flat
    # no per-gate line at the ceiling, but the informative tail is ranked
    assert "(its entire possible range)" not in text
    assert "gate #" in text
    assert "most load-bearing first" in text


def test_differing_ledger_keeps_the_ranking():
    """Defect 2, other branch: rows that differ are genuinely a ranking."""
    rng = np.random.default_rng(5)
    circ = random_circuit(rng, 4, 25)
    _, report = active_reduce(circ, ["Z0"])
    bounded = [
        r for r in report.kept_gate_ledger()
        if r["max_influence"] is not None
    ]
    caps = 2.0  # single Z observable: norm bound 1
    assert any(r["max_influence"] < caps for r in bounded), (
        "test circuit no longer produces differing rows; pick another seed"
    )
    text = report.explain()
    assert "most load-bearing first" in text
    assert "gate #" in text
    assert "could be bounded below the trivial maximum" not in text


@pytest.fixture()
def idle_qubit_case():
    # qubit 3 carries nothing any measurement can see
    c = Circuit(4).add("h", 0).add("cx", (0, 1)).add("cx", (1, 2)).add(
        "rz", 3, 0.7
    )
    return c


def test_qubit_drop_wording_compact_true(idle_qubit_case):
    """Defect 3: compact=True really drops -> current wording stands."""
    _, report = active_reduce(idle_qubit_case, ["Z0*Z1"], compact=True)
    assert report.removed_qubits == [2, 3]
    text = report.explain()
    assert "were dropped" in text
    assert "records the renumbering" in text
    assert "kept in place" not in text


def test_qubit_drop_wording_compact_false(idle_qubit_case):
    """Defect 3: compact=False drops nothing -> the text must not claim
    a drop or a renumbering that never happened."""
    _, report = active_reduce(idle_qubit_case, ["Z0*Z1"], compact=False)
    assert report.removed_qubits == [2, 3]
    assert report.output_metrics["qubits"] == 4
    text = report.explain()
    assert "were dropped" not in text
    assert "renumbering" not in text
    assert "kept in place because compact=False" in text
    assert "--no-compact" in text
