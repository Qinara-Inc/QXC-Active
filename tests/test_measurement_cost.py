"""Measurement cost and estimated hardware impact reporting."""

from __future__ import annotations

import pytest

from qxc_active import Circuit, active_reduce
from qxc_active.errors import InvalidOptionError


def ghz(n: int) -> Circuit:
    c = Circuit(n).add("h", 0)
    for q in range(n - 1):
        c.add("cx", (q, q + 1))
    return c


def test_measurement_cost_reported_and_unchanged():
    _, rep = active_reduce(ghz(6), ["Z0", "0.5*Z1*Z2 + 0.25*X3"])
    mc = rep.measurement_cost
    assert mc["terms_before"] == 3
    assert mc["terms_after"] == 3
    assert "measurement cost : 3 terms -> 3 terms (unchanged)" in rep.summary()


def test_hardware_impact_block_present_only_with_rates():
    _, rep = active_reduce(ghz(8), ["Z0"])
    assert rep.hardware_impact is None
    assert "hardware impact" not in rep.summary()

    rates = {"1q": 1e-4, "2q": 5e-3}
    _, rep = active_reduce(ghz(8), ["Z0"], error_rates=rates)
    hw = rep.hardware_impact
    assert hw is not None
    # GHZ measured with Z0: reduces to a single h... whatever remains, the
    # estimate must reflect fewer error-prone gates
    assert hw["expected_infidelity_after"] <= hw["expected_infidelity_before"]
    assert hw["model"] == "product_of_fidelities_estimate"
    out = rep.summary()
    assert "estimated hardware impact" in out
    assert "ESTIMATE" in out  # visually labelled, never conflated
    assert "certified ideal-value shift" in out


def test_infidelity_matches_product_model():
    # 1 h + 7 cx, nothing removable when measuring all qubits via X string
    circ = ghz(4)
    rates = {"1q": 1e-3, "2q": 1e-2}
    _, rep = active_reduce(circ, ["X0*X1*X2*X3"], error_rates=rates)
    hw = rep.hardware_impact
    expect_before = 1 - (1 - 1e-3) ** 1 * (1 - 1e-2) ** 3
    assert hw["expected_infidelity_before"] == pytest.approx(expect_before)


def diagonal_deep(n: int, n_1q: int, n_2q: int) -> Circuit:
    """Deep diagonal circuit: all gates commute with Z observables, so
    measuring Z0 removes everything — the saturated-infidelity regime."""
    c = Circuit(n)
    for i in range(n_1q):
        c.add("rz", i % n, 0.1 + 0.001 * i)
    for i in range(n_2q):
        c.add("cp", (i % n, (i + 1) % n), 0.2 + 0.001 * i)
    return c


def test_zero_gate_reduction_reports_positive_zero():
    # -expm1(0.0) is -0.0 in IEEE 754; the report must never carry it
    circ = diagonal_deep(6, 20, 30)
    reduced, rep = active_reduce(
        circ, ["Z0"], error_rates={"1q": 1e-4, "2q": 5e-3}
    )
    assert reduced.n_gates == 0
    assert repr(rep.hardware_impact["expected_infidelity_after"]) == "0.0"
    assert "-0.0" not in rep.summary()
    import json

    assert '"expected_infidelity_after": -0.0' not in rep.to_json()
    del json


def test_saturated_regime_reports_fidelity_with_factor():
    # infidelity saturates near 1.0 on deep circuits; the human line must
    # switch to fidelity, which still resolves the improvement
    circ = diagonal_deep(8, 100, 600)  # infid_before ~ 0.95
    reduced, rep = active_reduce(
        circ, ["Z0"], error_rates={"1q": 1e-4, "2q": 5e-3}
    )
    assert reduced.n_gates == 0
    hw = rep.hardware_impact
    assert hw["expected_infidelity_before"] > 0.9
    assert hw["expected_fidelity_after"] == 1.0
    assert hw["fidelity_improvement_factor"] == (
        1.0 / hw["expected_fidelity_before"]
    )
    out = rep.summary()
    assert "expected fidelity" in out
    assert "x better)" in out
    assert "expected infidelity" not in out
    # the ESTIMATE / proven divider must remain untouched
    assert "product-of-fidelities ESTIMATE" in out
    assert "certified ideal-value shift" in out


def test_low_error_regime_reports_infidelity_with_factor():
    # shallow circuit: infidelity is the natural quantity and resolves
    _, rep = active_reduce(
        ghz(8), ["Z0"], error_rates={"1q": 1e-4, "2q": 5e-3}
    )
    hw = rep.hardware_impact
    assert hw["expected_infidelity_before"] < 0.9
    assert hw["expected_infidelity_after"] > 0.0
    out = rep.summary()
    assert "expected infidelity" in out
    assert "x lower)" in out
    assert "expected fidelity" not in out
    # JSON carries everything regardless of what the human line chose
    for key in (
        "expected_infidelity_before", "expected_infidelity_after",
        "expected_fidelity_before", "expected_fidelity_after",
        "fidelity_improvement_factor",
    ):
        assert key in hw


def test_summary_leads_with_hardware_deciding_metrics():
    # two-qubit gates, depth, active qubits, measurement cost first;
    # total gate count stays but comes below
    _, rep = active_reduce(ghz(8), ["Z0"])
    out = rep.summary()

    def pos(marker: str) -> int:
        assert marker in out, marker
        return out.index(marker)

    assert (
        pos("2q+")
        < pos("depth")
        < pos("qubits")
        < pos("measurement cost")
        < pos("gates  :")
    )


def test_error_rate_validation():
    with pytest.raises(InvalidOptionError):
        active_reduce(ghz(3), ["Z0"], error_rates={"4q": 0.1})
    with pytest.raises(InvalidOptionError):
        active_reduce(ghz(3), ["Z0"], error_rates={"1q": 1.5})
    with pytest.raises(InvalidOptionError):
        # 2q gates present but no 2q rate: no invented defaults
        active_reduce(ghz(3), ["Z0*Z1*Z2"], error_rates={"1q": 1e-4})


def test_cli_error_rates_flag(tmp_path, capsys):
    from qxc_active.cli import main

    f = tmp_path / "c.qasm"
    f.write_text(
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\n'
        "h q[0];\ncx q[0], q[1];\ncx q[1], q[2];\n",
        encoding="utf-8",
    )
    code = main(
        [
            "reduce", str(f), "--observables", "Z0",
            "--error-rates", "1q=1e-4,2q=5e-3",
        ]
    )
    out = capsys.readouterr().out
    assert "estimated hardware impact" in out
    assert code == 0

    code = main(
        ["reduce", str(f), "--observables", "Z0", "--error-rates", "junk"]
    )
    assert code == 2
