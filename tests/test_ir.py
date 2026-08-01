"""Circuit IR: construction, metrics, equality, validation."""

import pytest

from qxc_active import Circuit, Gate, InvalidCircuitError


def test_builder_chaining():
    c = Circuit(2).add("h", 0).add("cx", (0, 1)).add("rz", 1, 0.5)
    assert c.n_gates == 3
    assert c.gates[1].qubits == (0, 1)
    assert c.gates[2].params == (0.5,)


def test_depth():
    c = Circuit(3).add("h", 0).add("h", 1).add("cx", (0, 1)).add("h", 2)
    assert c.depth() == 2
    assert Circuit(2).depth() == 0


def test_two_qubit_count():
    c = Circuit(3).add("h", 0).add("cx", (0, 1)).add("ccx", (0, 1, 2))
    assert c.two_qubit_count() == 2


def test_used_qubits():
    c = Circuit(5).add("h", 1).add("cx", (1, 3))
    assert c.used_qubits() == {1, 3}


def test_equality():
    a = Circuit(2).add("h", 0).add("rz", 1, 0.5)
    b = Circuit(2).add("h", 0).add("rz", 1, 0.5)
    c = Circuit(2).add("h", 0).add("rz", 1, 0.6)
    assert a == b and a != c


def test_copy_is_independent():
    a = Circuit(2).add("h", 0)
    b = a.copy()
    b.add("x", 1)
    assert a.n_gates == 1 and b.n_gates == 2


def test_negative_qubit_count_rejected():
    with pytest.raises(InvalidCircuitError):
        Circuit(-1)


def test_gate_out_of_range_rejected():
    with pytest.raises(InvalidCircuitError):
        Circuit(2).add("h", 2)


def test_non_gate_append_rejected():
    with pytest.raises(InvalidCircuitError):
        Circuit(1).append("h")  # type: ignore[arg-type]


def test_gate_str_and_circuit_str():
    c = Circuit(2).add("rz", 0, 0.5).add("cx", (0, 1))
    assert "rz(0.5) 0" in str(c)
    assert "Circuit(2 qubits, 2 gates)" in str(c)


def test_gate_immutable():
    g = Gate("h", (0,))
    with pytest.raises(Exception):
        g.name = "x"  # type: ignore[misc]
