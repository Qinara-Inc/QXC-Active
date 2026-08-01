"""Pauli string / observable parsing and algebra."""

import pytest

from qxc_active import InvalidObservableError, Observable, PauliTerm


class TestPauliTermParse:
    def test_sparse_single(self):
        t = PauliTerm.parse("Z0")
        assert t.axis(0) == "Z" and t.support_qubits() == [0]

    def test_sparse_multi(self):
        t = PauliTerm.parse("X1*Z4")
        assert t.axis(1) == "X" and t.axis(4) == "Z"
        assert t.support_qubits() == [1, 4]

    def test_sparse_whitespace_separator(self):
        assert PauliTerm.parse("Y2 Z3") == PauliTerm.parse("Y2*Z3")

    def test_dense(self):
        t = PauliTerm.parse("IZXY")
        assert t.axis(0) == "I" and t.axis(1) == "Z"
        assert t.axis(2) == "X" and t.axis(3) == "Y"

    def test_dense_leftmost_is_qubit0(self):
        assert PauliTerm.parse("ZI") == PauliTerm.parse("Z0")

    def test_single_letter_dense_identity(self):
        assert PauliTerm.parse("I").is_identity()

    def test_roundtrip(self):
        for s in ["Z0", "X1*Z4", "Y0*Y5"]:
            assert PauliTerm.parse(PauliTerm.parse(s).to_sparse()) == PauliTerm.parse(s)

    def test_to_dense(self):
        assert PauliTerm.parse("Z1").to_dense(3) == "IZI"

    def test_errors(self):
        for bad in ["", "Q0", "Z0*Z0", "Z-1", "Z0*", "*"]:
            with pytest.raises(InvalidObservableError):
                PauliTerm.parse(bad)

    def test_dense_ambiguity_prefers_sparse(self):
        # "X1" is sparse (X on qubit 1), not dense
        t = PauliTerm.parse("X1")
        assert t.support_qubits() == [1]


class TestObservable:
    def test_parse_sum(self):
        o = Observable.parse("0.5*Z0*Z1 + 0.25*X2 - Y0")
        coeffs = {t.to_sparse(): c for c, t in o.terms}
        assert coeffs == {"Z0*Z1": 0.5, "X2": 0.25, "Y0": -1.0}

    def test_parse_scientific_notation(self):
        o = Observable.parse("1e-3*Z0 + 2E-2*X1")
        coeffs = {t.to_sparse(): c for c, t in o.terms}
        assert coeffs == {"Z0": 1e-3, "X1": 2e-2}

    def test_merging_and_zero_drop(self):
        o = Observable.parse("Z0 + Z0 - 2*Z0 + X1")
        assert len(o) == 1 and o.terms[0][1].to_sparse() == "X1"

    def test_norm_bound(self):
        assert Observable.parse("0.5*Z0 - 1.5*X1").norm_bound() == 2.0

    def test_leading_minus(self):
        o = Observable.parse("-Z0")
        assert o.terms[0][0] == -1.0

    def test_coerce(self):
        assert isinstance(Observable.coerce("Z0"), Observable)
        assert isinstance(Observable.coerce(PauliTerm.parse("Z0")), Observable)
        with pytest.raises(InvalidObservableError):
            Observable.coerce(42)  # type: ignore[arg-type]

    def test_errors(self):
        for bad in ["", "0.5", "+ +", "0.5*"]:
            with pytest.raises(InvalidObservableError):
                Observable.parse(bad)

    def test_support_and_max_qubit(self):
        o = Observable.parse("Z0*X3 + Y1")
        assert o.max_qubit() == 3
        assert o.support() == 0b1011
