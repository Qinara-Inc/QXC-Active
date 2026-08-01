# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Per-observable back-propagation state.

A :class:`TermTracker` holds the back-propagated form of one observable
as a real-weighted Pauli-sum, walking backward through the circuit. It
supports exact conjugation through a gate, provable deviation bounds for
hypothetical gate removals, and two sound degradation mechanisms for
when the term count explodes on non-Clifford-dense circuits:

* **coarsening** (exact mode): collapse to support-only ("lightcone")
  tracking — still sound, merely less aggressive;
* **truncation** (approximate mode): drop the smallest-|coefficient|
  terms and charge their summed magnitude — a provable bound on the
  induced expectation-value change — against the error budget.

Every floating-point crumb discarded anywhere accumulates into ``dust``,
so reported bounds stay honest upper bounds including numerical noise.

Performance: a hybrid representation. Terms start in a plain dict keyed
by ``(x_mask, z_mask)`` — measured fastest below ~1000 terms — and
promote one-way to parallel numpy arrays (int64 x/z masks + float64
coefficients) past :data:`PROMOTE_THRESHOLD`, where vectorised
conjugation wins by 3-6x. The vector path dedups with ``np.lexsort`` +
``np.add.reduceat``, never unordered or atomic accumulation, so results
are deterministic under any PYTHONHASHSEED. Promotion requires every
qubit index <= 62 (masks must fit int64); the dict path handles any
width. Results across the promotion boundary are equal up to float
summation order (see CHANGELOG.md); removal decisions were verified
identical on the development benchmark suite.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .ir import Gate
from .paulis import Observable
from .transfer import gate_table, iter_axis_codes

Key = tuple[int, int]  # (x_mask, z_mask)

# Coefficients whose magnitude falls below this after merging are dropped and
# charged to the tracker's numerical-dust accumulator.
COEFF_DUST = 1e-12

# Term count above which a tracker promotes to the vectorised (numpy)
# representation. Measured: dict wins below ~1000 terms, numpy above.
PROMOTE_THRESHOLD = 1000

# (x_bit << 1 | z_bit) -> axis code (I=0, X=1, Y=2, Z=3)
_CODE_LUT = np.array([0, 3, 1, 2], dtype=np.int64)


@dataclass
class GatePlan:
    """The exactly-computed effect of one gate on one tracker.

    ``delta`` is a provable upper bound on the expectation-value change if
    the gate were *removed* at this point (math.inf when the tracker is
    coarse and overlaps the gate, i.e. no bound is available).
    """

    overlap: bool
    delta: float
    # (key, coeff, table entry) for each overlapping term; None when coarse
    # or vectorised. A table entry is (patterns, dust, deviation).
    entries: list[tuple[Key, float, tuple]] | None = None
    # vectorised-path payload: (selected indices, combined codes, table
    # vec_arrays). None on the dict/coarse paths.
    vec: tuple | None = None


@dataclass
class TermTracker:
    """Back-propagated form of one observable (fine terms or coarse support)."""

    name: str
    norm_bound: float  # sum |c| of the ORIGINAL observable (state-rewrite bound)
    fine: bool = True
    terms: dict[Key, float] = field(default_factory=dict)
    support: int = 0  # union support bit mask (over-approximation in fine mode)
    dust: float = 0.0  # accumulated numerical-noise bound
    max_terms_seen: int = 0
    coarsened: bool = False
    # hybrid representation: False = dict, True = parallel numpy arrays.
    # Promotion is one-way and only allowed when qubit indices fit int64.
    vec: bool = False
    allow_promotion: bool = False
    vxs: np.ndarray | None = None
    vzs: np.ndarray | None = None
    vcs: np.ndarray | None = None
    # commutation-aware coarse fallback: the ORIGINAL observable terms
    # (|coeff|, x, z) and every gate kept so far in this walk (with a
    # per-qubit index). Gate references only — no term lists, so this
    # cannot blow up, which is the whole reason coarse mode exists.
    seed: list[tuple[float, int, int]] = field(default_factory=list)
    kept_gates: list[Gate] = field(default_factory=list)
    kept_index: dict[int, list[int]] = field(default_factory=dict)
    # pairwise ||[H,G]|| function, injected by backward_pass (defining it
    # here would import passes and create a cycle)
    commutator_fn: object = None

    @staticmethod
    def from_observable(obs: Observable, name: str) -> "TermTracker":
        tr = TermTracker(name=name, norm_bound=obs.norm_bound())
        for coeff, term in obs.terms:
            tr._insert((term.x, term.z), coeff)
            tr.seed.append((abs(coeff), term.x, term.z))
        tr.max_terms_seen = len(tr.terms)
        return tr

    # -- term bookkeeping ---------------------------------------------------

    def _insert(self, key: Key, coeff: float) -> None:
        existing = self.terms.get(key)
        if existing is not None:
            new = existing + coeff
            if abs(new) <= COEFF_DUST:
                del self.terms[key]
                self.dust += abs(new)
            else:
                self.terms[key] = new
            return
        if abs(coeff) <= COEFF_DUST:
            self.dust += abs(coeff)
            return
        self.terms[key] = coeff
        self.support |= key[0] | key[1]

    def _refresh_support(self) -> None:
        if self.vec:
            if len(self.vxs):
                self.support = int(
                    np.bitwise_or.reduce(self.vxs | self.vzs)
                )
            else:
                self.support = 0
            return
        s = 0
        for x, z in self.terms:
            s |= x | z
        self.support = s

    def _maybe_promote(self) -> None:
        """One-way switch to the vectorised representation.

        The arrays are lexsorted into canonical (x, z) order, so the
        vectorised state is a pure function of the term *set* — dict
        insertion history does not leak into later arithmetic.
        """
        if (
            self.vec
            or not self.allow_promotion
            or len(self.terms) <= PROMOTE_THRESHOLD
        ):
            return
        xs = np.fromiter(
            (k[0] for k in self.terms), dtype=np.int64, count=len(self.terms)
        )
        zs = np.fromiter(
            (k[1] for k in self.terms), dtype=np.int64, count=len(self.terms)
        )
        cs = np.fromiter(
            self.terms.values(), dtype=np.float64, count=len(self.terms)
        )
        order = np.lexsort((zs, xs))
        self.vxs, self.vzs, self.vcs = xs[order], zs[order], cs[order]
        self.vec = True
        self.terms = {}

    def overlapping(self, gate_mask: int) -> list[Key]:
        """Terms whose support intersects the gate's qubit mask (dict mode)."""
        return [k for k in self.terms if (k[0] | k[1]) & gate_mask]

    # -- gate handling ------------------------------------------------------

    def plan_gate(self, gate: Gate) -> GatePlan:
        """Compute the gate's exact effect and the removal-cost bound."""
        gate_mask = 0
        for q in gate.qubits:
            gate_mask |= 1 << q
        if not self.fine:
            if not (self.support & gate_mask):
                return GatePlan(overlap=False, delta=0.0)
            # coarse mode used to give up here (delta = inf whenever the
            # gate touched the support mask). Qiskit's LightCone keeps
            # testing commutation in exactly this regime, which made it
            # strictly stronger; the commutation bound below closes that
            # gap while never keeping a gate the support test would have
            # removed.
            return GatePlan(
                overlap=True,
                delta=self._coarse_commutation_bound(gate, gate_mask),
            )
        if self.vec:
            return self._plan_gate_vec(gate, gate_mask)
        keys = self.overlapping(gate_mask)
        if not keys:
            return GatePlan(overlap=False, delta=0.0)
        table = gate_table(gate.unitary(), gate.cache_key(), gate.qubits)
        lookup = table.lookup
        terms = self.terms
        qubits = gate.qubits
        entries: list[tuple[Key, float, tuple]] = []
        delta = 0.0
        for key in sorted(keys):  # sorted for determinism
            coeff = terms[key]
            ent = lookup(iter_axis_codes(key[0], key[1], qubits))
            entries.append((key, coeff, ent))
            delta += abs(coeff) * ent[2]
        return GatePlan(overlap=True, delta=delta, entries=entries)

    def _plan_gate_vec(self, gate: Gate, gate_mask: int) -> GatePlan:
        sel = np.nonzero(
            (self.vxs | self.vzs) & np.int64(gate_mask)
        )[0]
        if not len(sel):
            return GatePlan(overlap=False, delta=0.0)
        table = gate_table(gate.unitary(), gate.cache_key(), gate.qubits)
        vtab = table.vec_arrays()
        codes = np.zeros(len(sel), dtype=np.int64)
        xs, zs = self.vxs[sel], self.vzs[sel]
        for j, q in enumerate(gate.qubits):
            xb = (xs >> np.int64(q)) & 1
            zb = (zs >> np.int64(q)) & 1
            codes |= _CODE_LUT[(xb << 1) | zb] << np.int64(2 * j)
        dev = vtab[5]
        delta = float(np.dot(np.abs(self.vcs[sel]), dev[codes]))
        return GatePlan(
            overlap=True, delta=delta, vec=(sel, codes, vtab)
        )

    # Cost bounds above this are useless to every removal path (dust
    # tolerance is capped at 1e-6), so the scan stops early and reports
    # "no usable bound". Purely a compute guard, never a soundness one.
    _COARSE_CUTOFF = 1e-6

    def _coarse_commutation_bound(self, gate: Gate, gate_mask: int) -> float:
        """Provable removal-cost bound for a gate in COARSE mode.

        Soundness (the same slide accounting the cancellation pass uses,
        so nothing compounds): write the remaining circuit as
        ``U_after · G · U_before`` with ``U_after`` the gates already
        walked and KEPT. Slide G to the measurement end one swap at a
        time; each swap with a kept gate H is a whole-unitary rewrite of
        norm ``||[H, G]||``, moving any expectation by at most
        ``2 * ||O|| * ||[H, G]||`` (passes.py, fact 2). Kept gates
        sharing no qubit with G commute exactly and swap for free.
        Once G sits next to the measurement, removing it costs at most
        ``||G' O G - O|| <= sum_j |c_j| * dev(G, P_j)`` over the ORIGINAL
        observable's terms (dev is the exact transfer deviation; for a
        Pauli wider than the gate, ``||[G, P_loc (x) P_rest]|| =
        ||[G, P_loc]||``). The bounds of successive rewrites add by the
        triangle inequality — LINEARLY, unlike a conjugation-drift
        argument, which is why the residuals may be charged the same way
        the cancellation pass charges its commutator slack.

        Removed gates are not in ``U_after`` and need no checking. When
        every residual is float-noise the total lands under
        dust_tolerance and the gate is removed as "invariant" with the
        residual charged to dust; a genuinely non-commuting gate drives
        the total past the cutoff and stays.
        """
        if self.commutator_fn is None:
            return math.inf
        total = 0.0
        table = None
        for weight, x, z in self.seed:
            if not ((x | z) & gate_mask):
                continue  # disjoint Pauli: commutes exactly
            if table is None:
                table = gate_table(
                    gate.unitary(), gate.cache_key(), gate.qubits
                )
            dev = table.lookup(iter_axis_codes(x, z, gate.qubits))[2]
            total += weight * dev
            if total > self._COARSE_CUTOFF:
                return math.inf
        candidates: set[int] = set()
        for q in gate.qubits:
            candidates.update(self.kept_index.get(q, ()))
        gate_qubits = set(gate.qubits)
        # most-recent kept gates first: a blocking neighbour fails fast
        for i in sorted(candidates, reverse=True):
            kept = self.kept_gates[i]
            if len(gate_qubits | set(kept.qubits)) > 6:
                return math.inf  # too large to analyse cheaply; keep
            residual = self.commutator_fn(kept, gate)
            total += 2.0 * self.norm_bound * residual
            if total > self._COARSE_CUTOFF:
                return math.inf
        return total

    def apply_plan(self, gate: Gate, plan: GatePlan) -> None:
        """Conjugate the tracked operator through the gate (gate is kept)."""
        # every kept gate joins the walk's cone (whatever the mode): if
        # the tracker coarsens later, the commutation fallback must test
        # against the full U_after, and a gate kept for ANOTHER
        # observable still blocks a slide for this one
        idx = len(self.kept_gates)
        self.kept_gates.append(gate)
        for q in gate.qubits:
            self.kept_index.setdefault(q, []).append(idx)
        gate_mask = 0
        for q in gate.qubits:
            gate_mask |= 1 << q
        if not self.fine:
            if plan.overlap:
                self.support |= gate_mask
            return
        if not plan.overlap:
            return
        if plan.vec is not None:
            self._apply_plan_vec(gate_mask, plan)
            return
        assert plan.entries is not None
        clear = ~gate_mask
        outputs: dict[Key, float] = {}
        out_get = outputs.get
        for key, coeff, (patterns, tdust, _dev) in plan.entries:
            # per-entry accumulation: keeps float addition order identical
            # to the reference implementation (byte-identical outputs)
            self.dust += abs(coeff) * tdust
            base_x, base_z = key[0] & clear, key[1] & clear
            for x_pat, z_pat, d in patterns:
                nk = (base_x | x_pat, base_z | z_pat)
                outputs[nk] = out_get(nk, 0.0) + coeff * d

        # Delete every input term, then insert every merged output. An
        # output key can never collide with a term the gate left untouched:
        # a traceless Pauli conjugates to a traceless operator, so the
        # all-identity component on the gate's own qubits always has
        # coefficient exactly zero — every output therefore still acts
        # non-trivially somewhere on the gate, while untouched terms act
        # trivially on all of it.
        terms = self.terms
        for key, _c, _t in plan.entries:
            del terms[key]
        for nk, val in outputs.items():
            self._insert(nk, val)
        self.max_terms_seen = max(self.max_terms_seen, len(terms))
        self._maybe_promote()
        # note: self.support is only an over-approximation in fine mode (it
        # grows monotonically); coarsen() recomputes it tightly on demand.

    def _apply_plan_vec(self, gate_mask: int, plan: GatePlan) -> None:
        sel, codes, vtab = plan.vec
        pat_x, pat_z, pat_d, offsets, counts, _dev, tdust = vtab
        cs_sel = self.vcs[sel]
        self.dust += float(np.dot(np.abs(cs_sel), tdust[codes]))

        lens = counts[codes]
        total = int(lens.sum())
        cum = np.cumsum(lens)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            cum - lens, lens
        )
        gidx = np.repeat(offsets[codes], lens) + within
        clear = np.int64(~gate_mask)
        out_x = np.repeat(self.vxs[sel] & clear, lens) | pat_x[gidx]
        out_z = np.repeat(self.vzs[sel] & clear, lens) | pat_z[gidx]
        out_c = np.repeat(cs_sel, lens) * pat_d[gidx]

        keep = np.ones(len(self.vxs), dtype=bool)
        keep[sel] = False
        # untouched terms cannot collide with outputs (same traceless-
        # conjugation argument as the dict path), so one global
        # lexsort + reduceat both dedups the outputs and restores the
        # canonical order — deterministic under any PYTHONHASHSEED.
        all_x = np.concatenate((self.vxs[keep], out_x))
        all_z = np.concatenate((self.vzs[keep], out_z))
        all_c = np.concatenate((self.vcs[keep], out_c))
        if len(all_x):
            order = np.lexsort((all_z, all_x))
            sx, sz, sc = all_x[order], all_z[order], all_c[order]
            starts_mask = np.empty(len(sx), dtype=bool)
            starts_mask[0] = True
            np.logical_or(
                sx[1:] != sx[:-1], sz[1:] != sz[:-1], out=starts_mask[1:]
            )
            starts = np.nonzero(starts_mask)[0]
            summed = np.add.reduceat(sc, starts)
            ux, uz = sx[starts], sz[starts]
            small = np.abs(summed) <= COEFF_DUST
            if small.any():
                self.dust += float(np.sum(np.abs(summed[small])))
                big = ~small
                ux, uz, summed = ux[big], uz[big], summed[big]
            self.vxs, self.vzs, self.vcs = ux, uz, summed
        else:
            self.vxs = np.empty(0, dtype=np.int64)
            self.vzs = np.empty(0, dtype=np.int64)
            self.vcs = np.empty(0, dtype=np.float64)
        if total:
            self.support |= int(np.bitwise_or.reduce(out_x | out_z))
        self.max_terms_seen = max(self.max_terms_seen, len(self.vxs))

    # -- degradation --------------------------------------------------------

    def coarsen(self) -> None:
        """Switch to sound support-only (lightcone-quality) tracking."""
        self._refresh_support()
        self.fine = False
        self.coarsened = True
        self.terms.clear()
        if self.vec:
            self.vec = False
            self.vxs = self.vzs = self.vcs = None

    def peek_truncation_cost(self, max_terms: int) -> float:
        """Cost bound of :meth:`truncate_to` without performing it."""
        excess = self.n_terms - max_terms
        if excess <= 0:
            return 0.0
        if self.vec:
            order = np.lexsort((self.vzs, self.vxs, np.abs(self.vcs)))
            return float(np.sum(np.abs(self.vcs[order[:excess]])))
        victims = heapq.nsmallest(
            excess, self.terms.items(), key=lambda kv: (abs(kv[1]), kv[0])
        )
        return sum(abs(c) for _, c in victims)

    def truncate_to(self, max_terms: int) -> float:
        """Drop smallest-|coeff| terms down to ``max_terms``.

        Returns the summed magnitude of dropped coefficients — a provable
        upper bound on the induced change of the final expectation value.
        Victim selection is by the total order (|coefficient|, x, z) in
        both representations.
        """
        excess = self.n_terms - max_terms
        if excess <= 0:
            return 0.0
        if self.vec:
            order = np.lexsort((self.vzs, self.vxs, np.abs(self.vcs)))
            victims = order[:excess]
            cost = float(np.sum(np.abs(self.vcs[victims])))
            keep = np.ones(len(self.vxs), dtype=bool)
            keep[victims] = False
            self.vxs = self.vxs[keep]
            self.vzs = self.vzs[keep]
            self.vcs = self.vcs[keep]
            self._refresh_support()
            return cost
        victims = heapq.nsmallest(
            excess, self.terms.items(), key=lambda kv: (abs(kv[1]), kv[0])
        )
        cost = 0.0
        for key, coeff in victims:
            cost += abs(coeff)
            del self.terms[key]
        self._refresh_support()
        return cost

    @property
    def n_terms(self) -> int:
        return len(self.vxs) if self.vec else len(self.terms)
