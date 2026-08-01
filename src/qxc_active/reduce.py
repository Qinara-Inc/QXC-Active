# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""The public entry point: :func:`active_reduce`.

Runs the three sound passes (backward observable propagation, commutation-
aware cancellation, initial-state simplification) to a fixpoint, then drops
qubits that no retained gate or observable touches. Approximate mode
(epsilon > 0) additionally removes low-influence gates under a proven
per-observable error budget, choosing the removal threshold by a small
deterministic ladder search (the certification never depends on the choice —
every candidate run enforces the budget exactly; the ladder only picks the
run that reduced the most).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Sequence

from . import __version__ as _pkg_version
from .errors import (
    BudgetViolationError,
    InvalidObservableError,
    InvalidOptionError,
)
from .fingerprint import subject_block
from .ir import Circuit, Gate
from .passes import (
    Item,
    PassState,
    backward_pass,
    cancellation_pass,
    clifford_resynthesis_pass,
    initial_state_pass,
)
from .paulis import (
    Distribution,
    Measurement,
    Observable,
    ObservableLike,
    PauliTerm,
    coerce_observables,
)
from .report import Report, RemovedGate, circuit_metrics

# Threshold-ladder divisors tried in approximate mode (theta = epsilon / d).
_THETA_DIVISORS = (1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)

logger = logging.getLogger("qxc_active")


@dataclass
class _RunResult:
    items: list[Item]
    state: PassState
    rounds: int
    timings_ms: dict[str, float]


def _run_pipeline(
    circuit: Circuit,
    observables: list[Observable],
    epsilon: float,
    theta: float,
    dust_tol: float,
    max_terms: int,
    lookback: int,
    max_rounds: int | None,
    clifford_resynthesis: bool = True,
    budget_groups: list[tuple[list[int], float]] | None = None,
) -> _RunResult:
    items: list[Item] = list(enumerate(circuit.gates))
    state = PassState(
        epsilon=epsilon,
        theta=theta,
        observables=observables,
        dust_tol=dust_tol,
        max_terms=max_terms,
        lookback=lookback,
        next_index=len(circuit.gates),
        budget_groups=budget_groups,
    )
    timings = {
        "backward": 0.0,
        "cancellation": 0.0,
        "initial_state": 0.0,
        "clifford_resynthesis": 0.0,
    }
    rounds = 0
    limit = max_rounds if max_rounds is not None else len(circuit.gates) + 2
    while rounds < max(limit, 1):
        rounds += 1
        t0 = time.perf_counter()
        ch1 = backward_pass(items, state)
        t1 = time.perf_counter()
        ch2 = cancellation_pass(items, state)
        t2 = time.perf_counter()
        ch3 = initial_state_pass(items, circuit.n_qubits, state)
        t3 = time.perf_counter()
        timings["backward"] += (t1 - t0) * 1e3
        timings["cancellation"] += (t2 - t1) * 1e3
        timings["initial_state"] += (t3 - t2) * 1e3
        logger.debug(
            "round %d (theta=%g): %d gates remain, %d removals so far",
            rounds, theta, len(items), len(state.removals),
        )
        if not (ch1 or ch2 or ch3):
            # only at a fixpoint: try the exact Clifford-prefix rewrite;
            # if it wins (strictly fewer gates), run the passes again over
            # the fresh prefix — total gate count strictly decreases, so
            # this terminates
            if clifford_resynthesis:
                t4 = time.perf_counter()
                improved = clifford_resynthesis_pass(
                    items, circuit.n_qubits, state
                )
                timings["clifford_resynthesis"] += (
                    time.perf_counter() - t4
                ) * 1e3
                if improved:
                    logger.debug(
                        "clifford resynthesis replaced the prefix; "
                        "%d gates remain",
                        len(items),
                    )
                    continue
            break
    for ev in state.events:
        logger.info(
            "observable %s: analysis %s at gate %d (term cap); "
            "result stays sound but less aggressive",
            ev["observable"], ev["action"], ev["at_gate"],
        )
    return _RunResult(items=items, state=state, rounds=rounds, timings_ms=timings)


def _remap_observable(obs: Measurement, qmap: dict[int, int]) -> Measurement:
    if isinstance(obs, Distribution):
        return Distribution(
            tuple(qmap[q] for q in obs.qubits), name=obs.name
        )
    terms = []
    for coeff, term in obs.terms:
        axes = {qmap[q]: term.axis(q) for q in term.support_qubits()}
        terms.append((coeff, PauliTerm.from_axes(axes)))
    return Observable(terms, name=obs.name)


def _expand_measurements(
    measurements: list[Measurement],
) -> tuple[list[Observable], list[tuple[list[int], float]]]:
    """Expand measurements into internal trackers and budget groups.

    An Observable becomes one tracker (singleton group, weight 1). A
    Distribution over qubits M becomes |M| trackers, one per generator Z_q,
    grouped with weight 1/2: any outcome probability changes by at most
    half the summed generator bounds (each outcome projector is a product
    of norm-<=1 factors (I +/- Z_q^H)/2, so factor changes of delta_q move
    the product by at most sum(delta_q)/2 — a proven per-outcome bound).
    """
    trackers: list[Observable] = []
    groups: list[tuple[list[int], float]] = []
    for m in measurements:
        if isinstance(m, Distribution):
            idxs = []
            for q in m.qubits:
                idxs.append(len(trackers))
                trackers.append(
                    Observable(
                        [(1.0, PauliTerm(0, 1 << q))],
                        name=f"{m.name or m}:Z{q}",
                    )
                )
            groups.append((idxs, 0.5))
        else:
            groups.append(([len(trackers)], 1.0))
            trackers.append(m)
    return trackers, groups


def _exact_value_on_zero_state(m: Measurement):
    """Exact value of a measurement on |0...0> — pure arithmetic, not a
    simulation. A Pauli term contributes its coefficient iff every
    factor is Z or I (any X or Y bit makes it vanish); a computational-
    basis measurement yields the all-zeros outcome with probability 1.
    """
    if isinstance(m, Distribution):
        return {"outcome": "0" * len(m.qubits), "probability": 1.0}
    # x-mask zero <=> no X or Y factor; + 0.0 normalises IEEE -0.0
    return sum(c for c, t in m.terms if t.x == 0) + 0.0


_ERROR_RATE_KEYS = ("1q", "2q", "3q")


def _validate_error_rates(error_rates: dict | None) -> dict | None:
    if error_rates is None:
        return None
    if not isinstance(error_rates, dict):
        raise InvalidOptionError(
            "error_rates must be a dict like {'1q': 1e-4, '2q': 5e-3}"
        )
    out = {}
    for k, v in error_rates.items():
        if k not in _ERROR_RATE_KEYS:
            raise InvalidOptionError(
                f"unknown error_rates key {k!r} (expected one of "
                f"{', '.join(_ERROR_RATE_KEYS)})"
            )
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise InvalidOptionError(f"error_rates[{k!r}] must be a number")
        v = float(v)
        if math.isnan(v) or not (0.0 <= v < 1.0):
            raise InvalidOptionError(
                f"error_rates[{k!r}] must be in [0, 1), got {v}"
            )
        out[k] = v
    return out


def _expected_fidelity(circuit: Circuit, rates: dict) -> tuple[float, float]:
    """(fidelity, infidelity) product-of-fidelities ESTIMATE, not a bound.

    Arity-1 gates use the '1q' rate; arity-k gates use the '{k}q' rate if
    given, else '2q' (a real device decomposes them into 2q gates anyway).
    A rate that is needed but not supplied is an error — no invented
    defaults.

    Both quantities are returned because each loses resolution where the
    other keeps it: infidelity saturates at 1.0 on deep circuits,
    fidelity at 1.0 on shallow ones. The ``+ 0.0`` normalises IEEE-754
    ``-0.0`` (an empty product gives ``-expm1(0.0) == -0.0``) so the
    reported value never reads as "negative zero".
    """
    log_fidelity = 0.0
    for gate in circuit.gates:
        k = len(gate.qubits)
        key = "1q" if k == 1 else (f"{k}q" if f"{k}q" in rates else "2q")
        if key not in rates:
            raise InvalidOptionError(
                f"error_rates has no {key!r} entry but the circuit "
                f"contains {k}-qubit gates; supply it explicitly "
                "(no defaults are invented)"
            )
        log_fidelity += math.log1p(-rates[key])
    return math.exp(log_fidelity), -math.expm1(log_fidelity) + 0.0


def _diagnose_no_reduction(
    circuit: Circuit,
    measurements: list[Measurement],
    state: PassState,
    epsilon: float,
    max_terms: int,
    dust_tolerance: float,
    cancellation_lookback: int,
) -> dict:
    """Explain WHY nothing was removable and suggest concrete next steps.

    Only called when the reduction removed nothing, so its (small) extra
    cost is paid exactly when a user is staring at an unhelpful result.
    """
    gate_support = 0
    for g in circuit.gates:
        for q in g.qubits:
            gate_support |= 1 << q
    obs_support = 0
    for m in measurements:
        obs_support |= m.support()
    covers_all = gate_support and (gate_support & ~obs_support) == 0

    events = list(state.events)
    peak = max(state.max_terms_seen, default=0)
    cap_reached = bool(events)
    cap_at_gate = events[0]["at_gate"] if events else None
    n_obs_qubits = obs_support.bit_count()

    if covers_all and cap_reached:
        why = (
            f"the observable acts on all {n_obs_qubits} qubits, so every "
            f"gate lies inside its lightcone, and term-level analysis "
            f"reached the cap of {max_terms:,} terms at gate {cap_at_gate} "
            "before it could prove anything commutes."
        )
    elif covers_all:
        why = (
            f"the observable acts on all {n_obs_qubits} qubits, so every "
            "gate lies inside its lightcone, and term-level analysis "
            "completed without proving any gate safe to remove."
        )
    elif cap_reached:
        why = (
            f"term-level analysis reached the cap of {max_terms:,} terms "
            f"at gate {cap_at_gate} before it could prove anything "
            "commutes."
        )
    else:
        why = (
            "term-level analysis completed without hitting any cap: no "
            "gate was proven to commute with what the observables become "
            "at its position, so none could be removed."
        )

    suggestions: list[dict] = []
    if cap_reached:
        suggestions.append(
            {
                "command": f"--max-terms {max_terms * 10}",
                "note": "the analysis was cut short, not exhausted",
            }
        )
    # concrete locality probe: what would a single Z0 achieve? (skipped
    # when the user's observable already is that local)
    if circuit.n_qubits >= 2 and (obs_support & ~1) != 0:
        try:
            probe = _run_pipeline(
                circuit,
                [Observable([(1.0, PauliTerm(0, 1))], name="Z0")],
                0.0,
                0.0,
                dust_tolerance,
                max_terms,
                cancellation_lookback,
                None,
            )
            probe_gates = len(probe.items)
            if probe_gates < circuit.n_gates:
                noun = "gate" if probe_gates == 1 else "gates"
                suggestions.append(
                    {
                        "command": "a more local observable",
                        "note": (
                            f"Z0 alone reduces this circuit to "
                            f"{probe_gates} {noun}"
                        ),
                    }
                )
        except Exception:  # the probe must never break the main path
            logger.debug("Z0 locality probe failed", exc_info=True)
    if epsilon == 0.0:
        suggestions.append(
            {
                "command": "--epsilon 0.01",
                "note": "allow a certified approximation",
            }
        )

    if not cap_reached:
        expected = (
            "This is a sound result rather than a failure: ACTIVE removes "
            "only gates it can prove are irrelevant, and here it could "
            "prove that for none of them. Their influence bounds are "
            "worst-case over all input states, so this is not evidence "
            "that every gate matters."
        )
    elif covers_all:
        expected = (
            "For QAOA-style ansätze measured with a global observable this "
            "is the expected result: at this budget the circuit has no "
            "removable gates."
        )
    else:
        expected = None

    return {
        "why": why,
        "suggestions": suggestions,
        "expected_note": expected,
        "peak_terms": peak,
        "max_terms": max_terms,
        "cap_reached": cap_reached,
        "cap_at_gate": cap_at_gate,
        "observable_covers_circuit": bool(covers_all),
    }


def active_reduce(
    circuit: Circuit,
    observables: Sequence[ObservableLike] | ObservableLike,
    epsilon: float = 0.0,
    *,
    max_terms: int = 20_000,
    cancellation_lookback: int = 100,
    max_rounds: int | None = None,
    dust_tolerance: float = 1e-12,
    compact: bool = True,
    verify: bool = False,
    clifford_resynthesis: bool = True,
    error_rates: dict | None = None,
) -> tuple[Circuit, Report]:
    """Reduce a circuit to only what affects the measured observables.

    Args:
        circuit: The input circuit (initial state |0...0>).
        observables: One or more Hermitian observables — ``Observable``
            objects, ``PauliTerm`` objects, or strings such as ``"Z0"``,
            ``"IZZ"`` or ``"0.5*Z0*Z1 + 0.25*X2"``.
        epsilon: Per-observable error budget. 0.0 (default) = exact mode:
            every rewrite provably preserves all expectation values (up to
            reported float dust ~1e-12). epsilon > 0 additionally removes
            low-influence gates; the report's ``certified_bounds`` are proven
            upper bounds on each |<O>_original - <O>_reduced| and the
            approximate part never exceeds epsilon.
        max_terms: Cap on back-propagated Pauli terms per observable before
            the analysis soundly degrades (coarsens to lightcone quality, or
            truncates within budget in approximate mode).
        cancellation_lookback: How far the cancellation pass scans backward
            over commuting gates for a cancellation partner.
        max_rounds: Optional cap on fixpoint rounds (default: until no pass
            changes anything).
        dust_tolerance: Magnitude below which float deviations are treated
            as numerical noise (still accounted in the reported dust bound).
        compact: If True (default), removed qubits are dropped and the
            remaining ones renumbered; ``report.qubit_map`` holds the
            old->new mapping and ``report.observables`` the observables
            remapped to the new numbering. If False, qubit indices are
            preserved (idle qubits stay in place).
        verify: If True, run the independent brute-force verifier
            (:mod:`qxc_active.verifier`) on the result and raise
            ``VerificationError`` if the claim does not hold. Exponential
            cost, only possible for small circuits — an explicit opt-in for
            critical jobs. Default False: zero overhead on the normal path.
        error_rates: Optional per-gate error rates, e.g.
            ``{"1q": 1e-4, "2q": 5e-3}``. When given, the report includes
            an *estimated* hardware impact (product-of-fidelities model of
            expected infidelity before vs after) — clearly labelled an
            estimate and kept strictly separate from the certified bound.
            When omitted, the block is omitted; no defaults are invented.
        clifford_resynthesis: If True (default), when the circuit begins
            with a block of Clifford gates, try replacing that prefix with
            a freshly synthesized preparation of the identical stabilizer
            state — an exact rewrite, kept only if strictly smaller. Costs
            a cheap name check on non-Clifford circuits.

    Returns:
        ``(reduced_circuit, report)``. The reduced circuit reproduces every
        observable's expectation value within ``report.certified_bounds``.

    Raises:
        InvalidCircuitError / InvalidObservableError / InvalidOptionError on
        malformed input. Valid input never raises.
    """
    if not isinstance(circuit, Circuit):
        raise InvalidOptionError(
            f"circuit must be a qxc_active.Circuit, got {type(circuit).__name__}"
        )
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise InvalidOptionError("epsilon must be a number")
    epsilon = float(epsilon)
    if math.isnan(epsilon) or math.isinf(epsilon) or epsilon < 0.0:
        raise InvalidOptionError(f"epsilon must be finite and >= 0, got {epsilon}")
    if max_terms < 16:
        raise InvalidOptionError(f"max_terms must be >= 16, got {max_terms}")
    if cancellation_lookback < 1:
        raise InvalidOptionError("cancellation_lookback must be >= 1")
    if not (0.0 <= dust_tolerance <= 1e-6):
        raise InvalidOptionError("dust_tolerance must be in [0, 1e-6]")
    if max_rounds is not None and max_rounds < 1:
        raise InvalidOptionError("max_rounds must be >= 1 (or None)")
    error_rates = _validate_error_rates(error_rates)

    measurements = coerce_observables(observables)
    for o in measurements:
        if o.max_qubit() >= circuit.n_qubits:
            raise InvalidObservableError(
                f"observable {o.name or o} acts on qubit {o.max_qubit()} but "
                f"the circuit has only {circuit.n_qubits} "
                f"qubit{'' if circuit.n_qubits == 1 else 's'}"
            )
    tracker_obs, budget_groups = _expand_measurements(measurements)

    t_start = time.perf_counter()

    def _one(theta: float) -> _RunResult:
        return _run_pipeline(
            circuit,
            tracker_obs,
            epsilon,
            theta,
            dust_tolerance,
            max_terms,
            cancellation_lookback,
            max_rounds,
            clifford_resynthesis=clifford_resynthesis,
            budget_groups=budget_groups,
        )

    if epsilon > 0.0:
        # deterministic threshold ladder; every run is independently
        # budget-certified, we simply keep the most effective one.
        best: _RunResult | None = None
        best_theta = 0.0
        best_key: tuple | None = None
        for div in _THETA_DIVISORS:
            theta = epsilon / div
            r = _one(theta)
            key = (len(r.items), sum(r.state.charges) + sum(r.state.dust))
            if best_key is None or key < best_key:
                best, best_key, best_theta = r, key, theta
        assert best is not None
        result, chosen_theta = best, best_theta
    else:
        result, chosen_theta = _one(0.0), 0.0

    state = result.state
    # budget invariant: enforced during the run; verified unconditionally
    # here (a real raise, not an assert, so the guarantee survives -O).
    for idxs, w in budget_groups:
        total = w * sum(state.charges[i] for i in idxs)
        if total > epsilon + 1e-15:
            raise BudgetViolationError(
                f"accumulated approximate charge {total!r} exceeds the "
                f"budget epsilon={epsilon!r}; this is a bug in qxc_active, "
                "not bad input — do not trust this reduction"
            )

    # ---- finalize: qubit elimination -------------------------------------
    used: set[int] = set()
    for _, g in result.items:
        used.update(g.qubits)
    for o in measurements:
        used.update(
            q for q in range(circuit.n_qubits) if (o.support() >> q) & 1
        )
    removed_qubits = sorted(set(range(circuit.n_qubits)) - used)

    if compact:
        kept = sorted(used)
        qmap = {old: new for new, old in enumerate(kept)}
        reduced = Circuit(len(kept))
        for _, g in result.items:
            reduced.append(
                Gate(g.name, tuple(qmap[q] for q in g.qubits), g.params, g.matrix)
            )
        out_observables = [_remap_observable(o, qmap) for o in measurements]
    else:
        qmap = {q: q for q in sorted(used)}
        reduced = Circuit(circuit.n_qubits)
        for _, g in result.items:
            reduced.append(g)
        out_observables = measurements

    total_ms = (time.perf_counter() - t_start) * 1e3
    timings = dict(result.timings_ms)
    timings["total"] = total_ms

    # ---- aggregate per-tracker numbers back to per-measurement -----------
    def _agg(values: list[float], idxs: list[int], w: float) -> float:
        return w * sum(values[i] for i in idxs)

    approx_bounds = [
        _agg(state.charges, idxs, w) for idxs, w in budget_groups
    ]
    numerical_dust = [_agg(state.dust, idxs, w) for idxs, w in budget_groups]
    max_terms_seen = [
        max(state.max_terms_seen[i] for i in idxs) for idxs, w in budget_groups
    ]

    def _agg_costs(costs: list[float]) -> list[float]:
        return [_agg(costs, idxs, w) for idxs, w in budget_groups]

    removed_records = [
        RemovedGate(
            index=r.index,
            name=r.gate.name,
            qubits=r.gate.qubits,
            reason=r.reason,
            cost_bound=max(_agg_costs(r.cost_per_observable), default=0.0),
            cost_per_observable=_agg_costs(r.cost_per_observable),
        )
        for r in sorted(state.removals, key=lambda r: r.index)
    ]
    influence_scores = {
        gate_idx: _agg_costs(deltas)
        for gate_idx, deltas in state.influence.items()
    }

    # an EMPTY reduced circuit is the best possible result: the register
    # stays in |0...0>, so every observable is classically determined by
    # exact arithmetic (a Pauli term contributes its coefficient iff it
    # has no X or Y factor — <0|Z|0> = 1, <0|X|0> = <0|Y|0> = 0) and no
    # quantum execution is needed at all. Say so, machine-readably.
    exact_values = None
    if reduced.n_gates == 0:
        exact_values = [
            _exact_value_on_zero_state(m) for m in out_observables
        ]

    # measurement cost: ACTIVE never grows the observable, unlike tools
    # that trim gates by growing it — worth stating explicitly. Counted
    # before vs after (after = the remapped observables; content is
    # unchanged by construction, only qubit numbering moves).
    def _terms_of(ms) -> int:
        return sum(
            len(m.terms) for m in ms if not isinstance(m, Distribution)
        )

    measurement_cost = {
        "terms_before": _terms_of(measurements),
        "terms_after": _terms_of(out_observables),
        "distributions": sum(
            1 for m in measurements if isinstance(m, Distribution)
        ),
    }

    hardware_impact = None
    if error_rates is not None:
        fid_before, infid_before = _expected_fidelity(circuit, error_rates)
        fid_after, infid_after = _expected_fidelity(reduced, error_rates)
        hardware_impact = {
            "error_rates": dict(sorted(error_rates.items())),
            "expected_infidelity_before": infid_before,
            "expected_infidelity_after": infid_after,
            "expected_fidelity_before": fid_before,
            "expected_fidelity_after": fid_after,
            # improvement as a fidelity ratio; None only when genuinely
            # undefined (fidelity_before underflowed to zero)
            "fidelity_improvement_factor": (
                fid_after / fid_before if fid_before > 0.0 else None
            ),
            "model": "product_of_fidelities_estimate",
        }

    subject = subject_block(
        circuit,
        measurements,
        {
            "epsilon": epsilon,
            "max_terms": max_terms,
            "cancellation_lookback": cancellation_lookback,
            "max_rounds": max_rounds,
            "dust_tolerance": dust_tolerance,
            "compact": compact,
            "clifford_resynthesis": clifford_resynthesis,
        },
    )

    diagnosis = None
    if not removed_records and not removed_qubits:
        diagnosis = _diagnose_no_reduction(
            circuit,
            measurements,
            state,
            epsilon,
            max_terms,
            dust_tolerance,
            cancellation_lookback,
        )

    report = Report(
        epsilon=epsilon,
        theta=chosen_theta,
        observable_names=[
            o.name or f"obs{i}" for i, o in enumerate(measurements)
        ],
        approx_bounds=approx_bounds,
        numerical_dust=numerical_dust,
        input_metrics=circuit_metrics(circuit),
        output_metrics=circuit_metrics(reduced),
        removed_gates=removed_records,
        removed_qubits=removed_qubits,
        qubit_map=qmap,
        observables=out_observables,
        influence_scores=influence_scores,
        coarsening_events=list(state.events),
        max_terms_seen=max_terms_seen,
        rounds=result.rounds,
        timings_ms=timings,
        version=_pkg_version,
        max_terms=max_terms,
        diagnosis=diagnosis,
        subject=subject,
        measurement_cost=measurement_cost,
        hardware_impact=hardware_impact,
        original_observables=list(measurements),
        exact_values=exact_values,
    )
    if verify:
        # explicit opt-in only; lazy import keeps the default path free of
        # even the module load
        from .verifier import verify_reduction

        verify_reduction(circuit, measurements, reduced, report)
    return reduced, report
