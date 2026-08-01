"""The zero-reduction output: named cause, concrete suggestions, no '-0%'."""

from __future__ import annotations

from pathlib import Path

from qxc_active import Circuit, active_reduce


def qaoa_ring(n: int, p: int) -> Circuit:
    """Canonical QAOA on a ring: h layer, then p x (rzz cost + rx mixer)."""
    c = Circuit(n)
    for q in range(n):
        c.add("h", q)
    for layer in range(p):
        gamma = 0.7 + 0.31 * layer
        beta = 0.4 + 0.17 * layer
        for q in range(n):
            c.add("rzz", (q, (q + 1) % n), gamma)
        for q in range(n):
            c.add("rx", q, beta)
    return c


GLOBAL_OBS = [" + ".join(f"Z{q}" for q in range(10))]


def test_qaoa_global_is_no_reduction_with_diagnosis():
    circ = qaoa_ring(10, 2)
    reduced, rep = active_reduce(circ, GLOBAL_OBS)
    assert rep.no_reduction
    assert reduced.n_gates == circ.n_gates
    d = rep.diagnosis
    assert d is not None
    assert d["observable_covers_circuit"]
    out = rep.summary()
    assert "No reduction possible" in out
    assert "shrank" not in out and "-0%" not in out
    # the Z0 probe produced a concrete number
    assert any("Z0 alone reduces" in s["note"] for s in d["suggestions"])
    # epsilon suggestion applies (exact mode was used)
    assert any(s["command"] == "--epsilon 0.01" for s in d["suggestions"])
    # cap was never approached, so --max-terms must NOT be suggested
    assert not d["cap_reached"]
    assert not any(
        s["command"].startswith("--max-terms") for s in d["suggestions"]
    )
    # states that the result is sound — while claiming only what is
    # established: nothing was provably removable, which is NOT the same
    # as every gate provably mattering
    assert "sound result rather than a failure" in out
    assert "not evidence that every gate matters" in out
    assert "provably influences" not in out
    assert "provably affects" not in out


def test_no_reduction_term_cap_variant():
    circ = qaoa_ring(10, 2)
    reduced, rep = active_reduce(circ, GLOBAL_OBS, max_terms=16)
    assert rep.no_reduction
    d = rep.diagnosis
    assert d["cap_reached"] and d["cap_at_gate"] is not None
    out = rep.summary()
    assert "cap reached" in out
    # now --max-terms DOES apply, with a concrete value
    assert any(
        s["command"] == "--max-terms 160" for s in d["suggestions"]
    )
    # explain() must lead with the conclusion, not the statistics
    exp = rep.explain()
    assert exp.startswith("No reduction possible")
    assert "shrank" not in exp


def test_no_reduction_epsilon_suggestion_omitted_in_approx_mode():
    circ = qaoa_ring(6, 1)
    obs = [" + ".join(f"Z{q}" for q in range(6))]
    _, rep = active_reduce(circ, obs, epsilon=1e-9)  # budget too small
    assert rep.no_reduction  # tiny epsilon: nothing affordable
    assert not any(
        s["command"] == "--epsilon 0.01"
        for s in rep.diagnosis["suggestions"]
    )


def test_reduction_case_keeps_normal_summary():
    c = Circuit(3).add("h", 0).add("cx", (0, 1)).add("cx", (1, 2))
    _, rep = active_reduce(c, ["Z0"])
    assert not rep.no_reduction
    assert rep.diagnosis is None
    assert "ACTIVE reduction" in rep.summary()
    assert "peak terms" in rep.summary()


def test_unbounded_kept_gates_summarized_once():
    # cap 16 forces coarsening: the many unbounded kept gates must appear
    # as ONE summary line, never one identical line per gate
    circ = qaoa_ring(10, 2)
    _, rep = active_reduce(circ, GLOBAL_OBS, max_terms=16)
    unbounded = [
        r for r in rep.kept_gate_ledger() if r["max_influence"] is None
    ]
    assert len(unbounded) > 5  # the flood the summary line prevents
    exp = rep.explain()
    assert (
        f"{len(unbounded)} gates: no bound available" in exp
    )
    assert "no bound available (analysis coarsened here)" not in exp


def test_probe_suggestion_uses_singular_for_one_gate():
    # h(1) + rx(0) measured globally: nothing removable, and the Z0 probe
    # reduces the circuit to exactly one gate — "1 gate", never "1 gates"
    c = Circuit(2).add("h", 1).add("rx", 0, 2.2)
    _, rep = active_reduce(c, ["Z0 + Z1"])
    assert rep.no_reduction
    notes = [s["note"] for s in rep.diagnosis["suggestions"]]
    assert any(n == "Z0 alone reduces this circuit to 1 gate" for n in notes)
    assert not any("1 gates" in n for n in notes)


def test_diagnosis_wording_claims_only_what_is_established():
    """A kept gate is "not proven safe to remove", never "proven to
    matter" — the influence bound is worst-case and can be vacuous."""
    # completed analysis, observable covers the circuit
    _, rep = active_reduce(qaoa_ring(10, 2), GLOBAL_OBS)
    why = rep.diagnosis["why"]
    assert "without proving any gate safe to remove" in why
    # completed analysis, observable covers only part of the circuit:
    # ring QAOA plus an rxx bridging to an uncovered 11th qubit (rxx acts
    # non-trivially on the |0> sector, so no pass can remove it)
    circ = qaoa_ring(10, 2)
    wide = Circuit(11)
    for g in circ.gates:
        wide.append(g)
    wide.add("rxx", (10, 9), 0.6)
    _, rep = active_reduce(wide, GLOBAL_OBS)
    assert rep.no_reduction and not rep.diagnosis["cap_reached"]
    assert not rep.diagnosis["observable_covers_circuit"]
    assert "no gate was proven to commute" in rep.diagnosis["why"]
    assert "so none could be removed" in rep.diagnosis["why"]


def test_no_overclaim_phrases_anywhere_in_src():
    """Permanent guard: "provably influences" / "provably affects" claim
    more than a worst-case bound establishes; they must never reappear
    in shipped code."""
    src = Path(__file__).parent.parent / "src" / "qxc_active"
    offenders = []
    for py in sorted(src.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for phrase in ("provably influences", "provably affects"):
            if phrase in text:
                offenders.append(f"{py.name}: {phrase!r}")
    assert not offenders, (
        "over-claiming phrases found (a kept gate is 'not proven safe to "
        f"remove', never 'proven to matter'): {offenders}"
    )


def test_json_report_carries_diagnosis():
    circ = qaoa_ring(10, 2)
    _, rep = active_reduce(circ, GLOBAL_OBS)
    blob = rep.to_dict()
    assert blob["no_reduction"] is True
    assert blob["diagnosis"]["why"]
    assert blob["max_terms"] == 20_000
