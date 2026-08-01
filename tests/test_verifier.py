"""The independent verifier: agrees with correct reductions, catches wrong
ones, stays entirely off the default product path."""

import subprocess
import sys

import numpy as np
import pytest

from conftest import random_circuit, random_observable
from qxc_active import Circuit, Gate, active_reduce
from qxc_active.paulis import coerce_observables
from qxc_active.verifier import (
    MAX_VERIFY_QUBITS,
    VerificationError,
    verify_reduction,
)


@pytest.mark.parametrize("seed", range(12))
def test_verifier_passes_correct_exact_reductions(seed):
    rng = np.random.default_rng(41000 + seed)
    n = int(rng.integers(2, 6))
    circ = random_circuit(rng, n, int(rng.integers(5, 30)))
    obs = coerce_observables([random_observable(rng, n)])
    reduced, report = active_reduce(circ, obs)
    result = verify_reduction(circ, obs, reduced, report)
    assert result.ok
    assert "PASS" in result.summary()


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("epsilon", [1e-3, 0.1])
def test_verifier_passes_correct_approx_reductions(seed, epsilon):
    rng = np.random.default_rng(42000 + seed)
    n = int(rng.integers(2, 6))
    circ = random_circuit(rng, n, int(rng.integers(5, 30)))
    obs = coerce_observables([random_observable(rng, n)])
    reduced, report = active_reduce(circ, obs, epsilon=epsilon)
    assert verify_reduction(circ, obs, reduced, report).ok


def test_verifier_catches_tampered_circuit():
    """Break a correct reduction and the verifier must scream."""
    circ = Circuit(2).add("h", 0).add("cx", (0, 1)).add("rz", 1, 0.7)
    obs = coerce_observables(["Z1"])
    reduced, report = active_reduce(circ, obs)
    # tamper: flip an angle in the reduced circuit
    bad = Circuit(reduced.n_qubits)
    for g in reduced.gates:
        if g.name == "h":
            bad.append(Gate("x", g.qubits))  # wrong gate
        else:
            bad.append(g)
    with pytest.raises(VerificationError, match="VIOLATION"):
        verify_reduction(circ, obs, bad, report)


def test_verifier_catches_dropped_needed_gate():
    circ = Circuit(2).add("h", 0).add("cx", (0, 1)).add("x", 1)
    obs = coerce_observables(["Z0*Z1"])
    reduced, report = active_reduce(circ, obs)
    if reduced.n_gates == 0:
        pytest.skip("nothing left to drop")
    bad = Circuit(reduced.n_qubits)
    for g in reduced.gates[:-1]:
        bad.append(g)
    result = verify_reduction(circ, obs, bad, report, raise_on_failure=False)
    assert not result.ok
    assert "FAIL" in result.summary()


def test_verifier_catches_understated_bound():
    """A report claiming a smaller bound than reality must fail."""
    # rz between the h's genuinely changes <Z0> (by 1 - cos 0.4 ~ 0.078)
    circ = Circuit(1).add("h", 0).add("rz", 0, 0.4).add("h", 0)
    obs = coerce_observables(["Z0"])
    reduced, report = active_reduce(circ, obs)
    assert any(g.name == "rz" for g in reduced.gates)  # it must be kept
    # forge a 'reduction' dropping the rz while keeping the exact-mode
    # (near-zero) certified bound from the honest report
    forged = Circuit(reduced.n_qubits)
    for g in reduced.gates:
        if g.name != "rz":
            forged.append(g)
    with pytest.raises(VerificationError):
        verify_reduction(circ, obs, forged, report)


def test_verifier_refuses_large_circuits():
    circ = Circuit(MAX_VERIFY_QUBITS + 1).add("h", 0)
    obs = coerce_observables(["Z0"])
    reduced, report = active_reduce(circ, obs)
    with pytest.raises(VerificationError, match="exponential"):
        verify_reduction(circ, obs, reduced, report)


def test_verify_flag_on_active_reduce():
    circ = Circuit(3).add("h", 0).add("cx", (0, 1)).add("cx", (1, 2))
    reduced, report = active_reduce(circ, ["Z0*Z1"], verify=True)
    assert reduced.n_gates == 0  # and no VerificationError raised


def test_verify_flag_approx_mode():
    rng = np.random.default_rng(7)
    circ = random_circuit(rng, 4, 20)
    active_reduce(circ, ["Z0"], epsilon=0.05, verify=True)


def test_default_path_never_loads_verifier():
    """ZERO overhead claim: a plain reduce must not even import the module."""
    code = (
        "import sys\n"
        "import qxc_active\n"
        "c = qxc_active.Circuit(3).add('h', 0).add('cx', (0, 1))\n"
        "qxc_active.active_reduce(c, ['Z1'], epsilon=0.01)\n"
        "assert 'qxc_active.verifier' not in sys.modules, 'verifier loaded!'\n"
        "print('clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_verifier_agrees_with_test_simulator():
    """Two independent second opinions must agree with each other."""
    from conftest import expectations

    rng = np.random.default_rng(11)
    for _ in range(5):
        n = int(rng.integers(2, 6))
        circ = random_circuit(rng, n, 15)
        obs = coerce_observables([random_observable(rng, n)])
        from qxc_active.verifier import _expectation

        assert _expectation(circ, obs[0]) == pytest.approx(
            expectations(circ, obs)[0], abs=1e-10
        )
