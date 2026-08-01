# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""The three sound reduction passes and their shared budget accounting.

All passes operate on ``items``: a list of ``(original_index, Gate)`` pairs,
mutated in place. Every rewrite is charged with a *provable* upper bound on
the change it can cause to each observable's final expectation value:

* exact rewrites charge only numerical dust (float noise, ~1e-12 scale);
* approximate rewrites (epsilon > 0) charge real bounds against the budget.

The mathematical facts used, stated once here and relied on throughout:

1. Unitary invariance: conjugation preserves operator norms, so a change of
   norm delta introduced at *any* point of the back-propagation changes the
   final expectation by at most delta (hybrid/triangle-inequality argument).
2. For a unitary-level rewrite ``U -> V`` of the whole circuit,
   ``|<O>_U - <O>_V| <= 2 * ||O|| * ||U - V||``, and for observables given
   as Pauli sums, ``||O|| <= sum |coefficients|``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .caches import cache_put
from .gates import (
    MERGEABLE_ROTATIONS,
    SYMMETRIC_GATES,
    deviation_from_phase_identity,
    embed,
)
from .ir import Gate  # noqa: F401  (used by clifford_resynthesis_pass)
from .paulis import Observable
from .tracker import TermTracker

Item = tuple[int, Gate]


@dataclass
class RemovalRecord:
    """One removed (or merged-away) gate with its certified cost bound."""

    index: int
    gate: Gate
    reason: str
    cost_per_observable: list[float]


@dataclass
class PassState:
    """Accounting shared by all passes across all fixpoint rounds."""

    epsilon: float
    theta: float
    observables: list[Observable]
    dust_tol: float
    max_terms: int
    lookback: int
    norm_bounds: list[float] = field(init=False)
    charges: list[float] = field(init=False)  # approx charges per tracker
    dust: list[float] = field(init=False)  # numerical-noise bounds
    # budget constraints: (tracker indices, weight); the approximate budget
    # rule is weight * sum(charges over the indices) <= epsilon. A plain
    # observable is a singleton group with weight 1; a Distribution groups
    # its Z_q generator trackers with weight 1/2 (per-outcome probability
    # bound = half the summed generator bounds). Defaults to singletons.
    budget_groups: list[tuple[list[int], float]] | None = None
    removals: list[RemovalRecord] = field(default_factory=list)
    influence: dict[int, list[float]] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    # gates created by resynthesis get fresh indices from this counter
    # (>= original gate count, so they never collide with input indices)
    next_index: int = 0
    max_terms_seen: list[int] = field(init=False)
    coarsened: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.observables)
        self.norm_bounds = [o.norm_bound() for o in self.observables]
        self.charges = [0.0] * n
        self.dust = [0.0] * n
        self.max_terms_seen = [len(o) for o in self.observables]
        self.coarsened = [False] * n
        if self.budget_groups is None:
            self.budget_groups = [([i], 1.0) for i in range(n)]

    def can_afford(self, deltas: list[float]) -> bool:
        for idxs, w in self.budget_groups:
            total = sum(self.charges[i] + deltas[i] for i in idxs)
            if w * total > self.epsilon:
                return False
        return True

    def can_afford_single(self, tracker_i: int, extra: float) -> bool:
        for idxs, w in self.budget_groups:
            if tracker_i not in idxs:
                continue
            total = sum(self.charges[i] for i in idxs) + extra
            if w * total > self.epsilon:
                return False
        return True

    def charge(self, deltas: list[float]) -> None:
        for i, d in enumerate(deltas):
            self.charges[i] += d

    def charge_dust(self, per_obs: list[float]) -> None:
        for i, d in enumerate(per_obs):
            self.dust[i] += d

    def unitary_rewrite_dust(self, norm_change: float) -> list[float]:
        """Dust charge for a whole-unitary rewrite of norm ``norm_change``."""
        return [2.0 * nb * norm_change for nb in self.norm_bounds]


# ---------------------------------------------------------------------------
# Pass 1: backward observable propagation + dead-code elimination
# ---------------------------------------------------------------------------


def backward_pass(items: list[Item], state: PassState) -> bool:
    """Heisenberg back-propagation; removes irrelevant/invariant gates.

    Walks the circuit from the last gate to the first, dragging every
    observable backward as an exact Pauli-sum. Removes gates that (a) act
    outside every observable's current support, (b) leave every observable
    exactly invariant, or (c) in approximate mode, have influence <= theta
    affordable within the remaining budget.
    """
    trackers = [
        TermTracker.from_observable(o, o.name or f"obs{i}")
        for i, o in enumerate(state.observables)
    ]
    # the vectorised representation stores Pauli masks as int64: it is
    # only allowed when every qubit index (gates and observables) is <= 62
    max_q = -1
    for _, g in items:
        for q in g.qubits:
            if q > max_q:
                max_q = q
    obs_ok = all(o.support().bit_length() <= 63 for o in state.observables)
    vec_ok = max_q <= 62 and obs_ok
    for tr in trackers:
        tr.allow_promotion = vec_ok
        # the coarse-mode commutation fallback reuses this pass module's
        # cached pairwise commutator (injected to avoid an import cycle)
        tr.commutator_fn = _commutator_norm
    keep: list[Item] = []
    changed = False
    n_obs = len(trackers)

    for orig_idx, gate in reversed(items):
        plans = [tr.plan_gate(gate) for tr in trackers]
        if not any(p.overlap for p in plans):
            state.removals.append(
                RemovalRecord(orig_idx, gate, "outside_support", [0.0] * n_obs)
            )
            state.influence.pop(orig_idx, None)
            changed = True
            continue
        deltas = [p.delta for p in plans]
        if all(d <= state.dust_tol for d in deltas):
            state.charge_dust(deltas)
            state.removals.append(
                RemovalRecord(orig_idx, gate, "invariant", deltas)
            )
            state.influence.pop(orig_idx, None)
            changed = True
            continue
        if (
            state.theta > 0.0
            and all(d <= state.theta for d in deltas)
            and state.can_afford(deltas)
        ):
            state.charge(deltas)
            state.removals.append(
                RemovalRecord(orig_idx, gate, "low_influence", deltas)
            )
            state.influence.pop(orig_idx, None)
            changed = True
            continue
        # gate is kept: conjugate all observables through it
        state.influence[orig_idx] = deltas
        for tr, plan in zip(trackers, plans):
            tr.apply_plan(gate, plan)
        for i, tr in enumerate(trackers):
            state.max_terms_seen[i] = max(state.max_terms_seen[i], tr.max_terms_seen)
            if tr.fine and tr.n_terms > state.max_terms:
                _degrade(tr, i, orig_idx, state)
        keep.append((orig_idx, gate))

    keep.reverse()
    if changed:
        items[:] = keep
    for i, tr in enumerate(trackers):
        state.dust[i] += tr.dust
        state.coarsened[i] = state.coarsened[i] or tr.coarsened
    return changed


def _degrade(tr: TermTracker, obs_i: int, at_gate: int, state: PassState) -> None:
    """Handle a term-count blow-up soundly: truncate (approx) or coarsen."""
    if state.theta > 0.0:
        cost = tr.peek_truncation_cost(state.max_terms)
        if state.can_afford_single(obs_i, cost):
            actual = tr.truncate_to(state.max_terms)
            state.charges[obs_i] += actual
            state.events.append(
                {
                    "observable": tr.name,
                    "at_gate": at_gate,
                    "action": "truncated",
                    "cost_bound": actual,
                }
            )
            return
    tr.coarsen()
    state.coarsened[obs_i] = True
    state.events.append(
        {
            "observable": tr.name,
            "at_gate": at_gate,
            "action": "coarsened",
            "cost_bound": 0.0,
        }
    )


# ---------------------------------------------------------------------------
# Pass 2: commutation-aware cancellation and rotation merging
# ---------------------------------------------------------------------------

_PAIR_CACHE: dict[tuple, float] = {}
_IDENTITY_CACHE: dict[tuple, float] = {}
_PAIR_CACHE_LIMIT = 200_000


def _identity_deviation(gate: Gate) -> float:
    key = gate.cache_key()
    hit = _IDENTITY_CACHE.get(key)
    if hit is None:
        hit, _ = deviation_from_phase_identity(gate.unitary())
        cache_put(_IDENTITY_CACHE, key, hit, _PAIR_CACHE_LIMIT)
    return hit


def _union_and_embeds(p: Gate, g: Gate) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    union = tuple(sorted(set(p.qubits) | set(g.qubits)))
    a = embed(p.unitary(), p.qubits, union)
    b = embed(g.unitary(), g.qubits, union)
    return union, a, b


def _rel(qubits: tuple[int, ...], union: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(union.index(q) for q in qubits)


def _commutator_norm(p: Gate, g: Gate) -> float:
    union = tuple(sorted(set(p.qubits) | set(g.qubits)))
    key = (p.cache_key(), _rel(p.qubits, union), g.cache_key(), _rel(g.qubits, union))
    hit = _PAIR_CACHE.get(key)
    if hit is not None:
        return hit
    _, a, b = _union_and_embeds(p, g)
    val = float(np.linalg.norm(b @ a - a @ b, 2))
    cache_put(_PAIR_CACHE, key, val, _PAIR_CACHE_LIMIT)
    return val


def _mergeable(p: Gate, g: Gate) -> bool:
    if p.name != g.name or p.name not in MERGEABLE_ROTATIONS:
        return False
    if p.matrix is not None or g.matrix is not None:
        return False
    if p.qubits == g.qubits:
        return True
    return p.name in SYMMETRIC_GATES and set(p.qubits) == set(g.qubits)


def cancellation_pass(items: list[Item], state: PassState) -> bool:
    """Peephole cancellation with exact commutation-aware lookback.

    Removes gates that are (a phase multiple of) the identity, cancels
    adjacent-up-to-commutation inverse pairs, and merges rotations of the
    same family on the same qubits. All rewrites preserve the circuit
    unitary up to global phase; float slack is charged as numerical dust.
    """
    changed = False

    # single-gate identity sweep
    j = 0
    while j < len(items):
        idx, g = items[j]
        dev = _identity_deviation(g)
        if dev <= state.dust_tol:
            costs = state.unitary_rewrite_dust(dev)
            state.charge_dust(costs)
            state.removals.append(RemovalRecord(idx, g, "identity", costs))
            state.influence.pop(idx, None)
            del items[j]
            changed = True
        else:
            j += 1

    # pair cancellation / merging with commutation-aware backward scan
    i = 1
    while i < len(items):
        idx_g, g = items[i]
        g_qubits = set(g.qubits)
        eta = 0.0  # accumulated commutator slack along the scan
        fired = False
        j = i - 1
        steps = 0
        while j >= 0 and steps < state.lookback:
            idx_p, p = items[j]
            p_qubits = set(p.qubits)
            if not (p_qubits & g_qubits):
                j -= 1
                steps += 1
                continue
            if len(p_qubits | g_qubits) > 6:
                break  # too large to analyse cheaply; stay conservative
            if p_qubits == g_qubits:
                union, a, b = _union_and_embeds(p, g)
                prod = b @ a
                dev, _ = deviation_from_phase_identity(prod)
                if dev <= state.dust_tol:
                    costs = state.unitary_rewrite_dust(eta + dev)
                    state.charge_dust(costs)
                    state.removals.append(
                        RemovalRecord(idx_g, g, "cancelled", costs)
                    )
                    state.removals.append(
                        RemovalRecord(idx_p, p, "cancelled", [0.0] * len(costs))
                    )
                    state.influence.pop(idx_g, None)
                    state.influence.pop(idx_p, None)
                    del items[i]
                    del items[j]
                    fired = True
                    break
                if _mergeable(p, g):
                    merged = Gate(p.name, p.qubits, (p.params[0] + g.params[0],))
                    m_emb = embed(merged.unitary(), merged.qubits, union)
                    dev_m = float(np.linalg.norm(m_emb - prod, 2))
                    costs = state.unitary_rewrite_dust(eta + dev_m)
                    state.charge_dust(costs)
                    state.removals.append(
                        RemovalRecord(idx_g, g, "merged", costs)
                    )
                    state.influence.pop(idx_g, None)
                    del items[i]
                    dev_i = _identity_deviation(merged)
                    if dev_i <= state.dust_tol:
                        costs2 = state.unitary_rewrite_dust(dev_i)
                        state.charge_dust(costs2)
                        state.removals.append(
                            RemovalRecord(idx_p, p, "cancelled", costs2)
                        )
                        state.influence.pop(idx_p, None)
                        del items[j]
                    else:
                        items[j] = (idx_p, merged)
                    fired = True
                    break
            # p is not a partner; can g slide left past p?
            comm = _commutator_norm(p, g)
            if comm <= state.dust_tol:
                eta += comm
                j -= 1
                steps += 1
                continue
            break
        if fired:
            changed = True
            i = max(j, 1)
        else:
            i += 1
    return changed


# ---------------------------------------------------------------------------
# Pass 2b: targeted Clifford resynthesis (optional, exact, keep-if-smaller)
# ---------------------------------------------------------------------------


def clifford_resynthesis_pass(items: list[Item], n_qubits: int, state: PassState) -> bool:
    """Replace the maximal all-Clifford prefix with a freshly synthesized,
    strictly smaller preparation of the identical stabilizer state.

    Exact by construction (integer tableau; the synthesis hard-verifies its
    result and bails on any surprise), so nothing is charged to any bound.
    On non-Clifford prefixes the cost is a cheap name check.
    """
    from .clifford import (
        CliffordEngineError,
        clifford_prefix_length,
        simulate_prefix,
        synthesize_preparation,
    )

    k = clifford_prefix_length([g for _, g in items])
    if k < 4:  # too small for synthesis (~n^2 gates) to ever win
        return False
    prefix = [g for _, g in items[:k]]
    # restrict to the qubits the prefix touches: untouched qubits stay in
    # |0> and need no gates, and this keeps the synthesis (and therefore
    # idempotence) independent of idle qubits and of qubit renumbering
    touched = sorted({q for g in prefix for q in g.qubits})
    local_of = {q: i for i, q in enumerate(touched)}
    local_prefix = [
        Gate(g.name, tuple(local_of[q] for q in g.qubits), g.params)
        for g in prefix
    ]
    try:
        tableau = simulate_prefix(len(touched), local_prefix)
    except CliffordEngineError:
        # engine surprise: abort the resynthesis, keep the original prefix
        # (a lost optimization, never a lost guarantee)
        return False
    local_prep = synthesize_preparation(tableau)
    if local_prep is None or len(local_prep) >= k:
        return False
    prep = [
        Gate(g.name, tuple(touched[q] for q in g.qubits), g.params)
        for g in local_prep
    ]
    n_obs = len(state.observables)
    for idx, g in items[:k]:
        state.removals.append(
            RemovalRecord(idx, g, "clifford_resynthesis", [0.0] * n_obs)
        )
        state.influence.pop(idx, None)
    new_items: list[Item] = []
    for g in prep:
        new_items.append((state.next_index, g))
        state.next_index += 1
    items[:k] = new_items
    return True


# ---------------------------------------------------------------------------
# Pass 3: initial-state simplification
# ---------------------------------------------------------------------------


def initial_state_pass(items: list[Item], n_qubits: int, state: PassState) -> bool:
    """Remove gates acting trivially on the reachable |0..0>-sector states.

    Tracks which qubits are still provably in |0> as the circuit executes.
    A gate whose action on the subspace (|0> on those qubits, anything
    elsewhere) is a global phase is removed; a gate that preserves a qubit's
    |0> sector keeps that qubit in the known set.
    """
    known0 = set(range(n_qubits))
    changed = False
    j = 0
    while j < len(items):
        idx, g = items[j]
        in_known = [t for t, q in enumerate(g.qubits) if q in known0]
        if not in_known:
            j += 1
            continue
        u = g.unitary()
        k = len(g.qubits)
        # column/row selection: gate qubit t is bit (k-1-t), MSB-first
        a_mask = 0
        for t in in_known:
            a_mask |= 1 << (k - 1 - t)
        cols = [r for r in range(2**k) if not (r & a_mask)]
        m = u[:, cols]
        rows_out = [r for r in range(2**k) if r & a_mask]
        leak = float(np.linalg.norm(m[rows_out, :], 2)) if rows_out else 0.0
        s_block = m[cols, :]
        dev_id, _ = deviation_from_phase_identity(s_block)
        if dev_id + leak <= state.dust_tol:
            costs = state.unitary_rewrite_dust(dev_id + leak)
            state.charge_dust(costs)
            state.removals.append(
                RemovalRecord(idx, g, "trivial_on_initial_state", costs)
            )
            state.influence.pop(idx, None)
            del items[j]
            changed = True
            continue
        # gate stays: which of its known-|0> qubits remain provably |0>?
        for t in in_known:
            bit = 1 << (k - 1 - t)
            rows_a1 = [r for r in range(2**k) if r & bit]
            leak_a = float(np.linalg.norm(m[rows_a1, :], 2))
            if leak_a <= state.dust_tol:
                state.charge_dust(state.unitary_rewrite_dust(leak_a))
            else:
                known0.discard(g.qubits[t])
        j += 1
    return changed
