"""Verification simulator sanity checks against hand-computed values."""

import math

import numpy as np
import pytest

from qxc_active import Circuit, InvalidCircuitError, Observable
from qxc_active import sim


def test_zero_state():
    psi = sim.statevector(Circuit(2))
    assert np.allclose(psi, [1, 0, 0, 0])


def test_bell_state():
    c = Circuit(2).add("h", 0).add("cx", (0, 1))
    psi = sim.statevector(c)
    assert np.allclose(psi, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])


def test_bell_correlations():
    c = Circuit(2).add("h", 0).add("cx", (0, 1))
    assert sim.expectation(c, Observable.parse("Z0*Z1")) == pytest.approx(1.0)
    assert sim.expectation(c, Observable.parse("Z0")) == pytest.approx(0.0)
    assert sim.expectation(c, Observable.parse("X0*X1")) == pytest.approx(1.0)
    assert sim.expectation(c, Observable.parse("Y0*Y1")) == pytest.approx(-1.0)


def test_rotation_analytic():
    theta = 0.813
    c = Circuit(1).add("rx", 0, theta)
    assert sim.expectation(c, Observable.parse("Z0")) == pytest.approx(
        math.cos(theta)
    )
    assert sim.expectation(c, Observable.parse("Y0")) == pytest.approx(
        -math.sin(theta)
    )


def test_qubit_order_convention():
    # x on qubit 0 flips the MOST significant index bit
    c = Circuit(2).add("x", 0)
    psi = sim.statevector(c)
    assert abs(psi[0b10] - 1.0) < 1e-12


def test_weighted_sum_observable():
    c = Circuit(2).add("x", 0)
    o = Observable.parse("0.5*Z0 + 2*Z1")
    assert sim.expectation(c, o) == pytest.approx(0.5 * (-1) + 2 * 1)


def test_three_qubit_gate():
    c = Circuit(3).add("x", 0).add("x", 1).add("ccx", (0, 1, 2))
    assert sim.expectation(c, Observable.parse("Z2")) == pytest.approx(-1.0)


def test_qubit_limit():
    with pytest.raises(InvalidCircuitError):
        sim.statevector(Circuit(sim.MAX_SIM_QUBITS + 1))


def test_observable_out_of_range():
    with pytest.raises(InvalidCircuitError):
        sim.expectation(Circuit(1), Observable.parse("Z5"))
