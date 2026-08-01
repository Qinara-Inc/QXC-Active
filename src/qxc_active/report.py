# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Reduction report: what was removed, why, and with what certified bound."""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .ir import Circuit
from .paulis import Distribution, Measurement, Observable

# Plain-language description of every removal reason (used by explain()).
REASON_DESCRIPTIONS: dict[str, str] = {
    "outside_support": (
        "never touches anything the measured observables depend on"
    ),
    "invariant": (
        "commutes with what the observables have become at that point in "
        "the circuit (proven exactly)"
    ),
    "low_influence": (
        "has influence below the error budget; its worst-case effect was "
        "charged against epsilon"
    ),
    "identity": "equals the identity operation (up to a global phase)",
    "cancelled": "cancels exactly against a partner gate",
    "merged": "was merged into an adjacent rotation of the same family",
    "trivial_on_initial_state": (
        "acts trivially on the |0...0> state reachable at its position"
    ),
    "clifford_resynthesis": (
        "was part of a Clifford block that got replaced by a smaller, "
        "freshly synthesized preparation of the exact same state"
    ),
}


# Version of the report DICT/JSON format, independent of the package
# version. Bump on breaking changes to the report structure.
# 1.1: added the top-level "measurement" block (original + remapped
#      observables in canonical and structured form, qubit map as an
#      integer-pair list) — additive, nothing removed or renamed.
# 1.2: measurement-block entries gain "exact_value" when (and ONLY when)
#      the reduced circuit is empty — a float for observables, an
#      {outcome, probability} object for distributions. Absent whenever
#      gates remain, so "not computed" can never read as "zero".
SCHEMA_VERSION = "1.2"


def _plural(count: int, noun: str) -> str:
    """``1 gate`` / ``3 gates`` — no more ``1 gates`` in user output."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


def circuit_metrics(circuit: Circuit) -> dict[str, int]:
    """The four headline metrics of a circuit."""
    return {
        "qubits": circuit.n_qubits,
        "gates": circuit.n_gates,
        "depth": circuit.depth(),
        "two_qubit_gates": circuit.two_qubit_count(),
    }


@dataclass
class RemovedGate:
    """One gate removed (or merged away) during reduction."""

    index: int  # position in the ORIGINAL gate list
    name: str
    qubits: tuple[int, ...]
    reason: str  # outside_support | invariant | low_influence | identity |
    #              cancelled | merged | trivial_on_initial_state
    cost_bound: float  # max over observables of the certified cost bound
    cost_per_observable: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "qubits": list(self.qubits),
            "reason": self.reason,
            "cost_bound": self.cost_bound,
            "cost_per_observable": self.cost_per_observable,
        }


@dataclass
class Report:
    """Full account of one :func:`~qxc_active.reduce.active_reduce` run.

    Error-bound semantics (read carefully — this is the honest part):

    * ``approx_bounds[i]``: proven upper bound on the change of observable
      i's expectation value caused by *approximate* rewrites (low-influence
      gate removals and term truncations). Guaranteed <= epsilon. Exactly
      0.0 in exact mode.
    * ``numerical_dust[i]``: proven upper bound on the effect of every
      floating-point crumb discarded anywhere (typically ~1e-12 scale).
      This exists because IEEE-754 arithmetic is not exact; reporting it
      separately is more honest than calling exact mode "error zero".
    * ``certified_bounds[i] = approx_bounds[i] + numerical_dust[i]`` is the
      total proven upper bound on |<O_i>_original - <O_i>_reduced|.
    """

    epsilon: float
    theta: float
    observable_names: list[str]
    approx_bounds: list[float]
    numerical_dust: list[float]
    input_metrics: dict[str, int]
    output_metrics: dict[str, int]
    removed_gates: list[RemovedGate]
    removed_qubits: list[int]
    qubit_map: dict[int, int]  # original index -> reduced index (kept qubits)
    observables: list[Measurement]  # remapped to reduced qubit indices
    influence_scores: dict[int, list[float]]  # retained original gate idx -> per-obs
    coarsening_events: list[dict]
    max_terms_seen: list[int]
    rounds: int
    timings_ms: dict[str, float]
    version: str
    # the resolved term cap the run used (peak counts are shown against it)
    max_terms: int = 20_000
    # set only when the run removed nothing: why, and what to try next
    diagnosis: dict | None = None
    # what this certificate is about: stable hashes of the canonicalised
    # input circuit and each observable, plus the resolved options
    # (identical inputs -> identical hashes across runs, machines and
    # PYTHONHASHSEED values)
    subject: dict | None = None
    # observable size before/after (ACTIVE never grows it — the cost of
    # measuring stays exactly what it was)
    measurement_cost: dict | None = None
    # ESTIMATED hardware impact (product-of-fidelities model), present
    # only when the caller supplied error rates. An estimate, never to be
    # conflated with the certified bound.
    hardware_impact: dict | None = None
    # the measurements exactly as the caller supplied them (original
    # qubit numbering) — serialised in the "measurement" block so a
    # report consumer can see what was asked for
    original_observables: list | None = None
    # set ONLY when the reduced circuit is empty: the classically
    # determined value of each measurement on |0...0> (float for an
    # observable, {outcome, probability} for a distribution). None when
    # any gate remains — absence must never read as "computed as zero".
    exact_values: list | None = None

    @property
    def no_reduction(self) -> bool:
        """True when the run removed no gates and no qubits."""
        return not self.removed_gates and not self.removed_qubits

    @property
    def certified_bounds(self) -> list[float]:
        """Total proven bound per observable (approx charges + float dust)."""
        return [a + d for a, d in zip(self.approx_bounds, self.numerical_dust)]

    @property
    def n_gates_removed(self) -> int:
        return len(self.removed_gates)

    def removed_by_reason(self) -> dict[str, int]:
        """Removal counts keyed by reason."""
        out: dict[str, int] = {}
        for r in self.removed_gates:
            out[r.reason] = out.get(r.reason, 0) + 1
        return dict(sorted(out.items()))

    def kept_gate_ledger(self, top: int | None = None) -> list[dict[str, Any]]:
        """Retained gates with their influence, most influential first.

        ``max_influence`` is the proven worst-case effect on any measurement
        if that gate were removed (None = unbounded because the analysis
        coarsened there).
        """
        rows = []
        for idx, scores in self.influence_scores.items():
            finite = [s for s in scores if math.isfinite(s)]
            mx: float | None
            if len(finite) < len(scores):
                mx = None  # at least one measurement has no bound (coarse)
            else:
                mx = max(finite, default=0.0)
            rows.append(
                {
                    "index": idx,
                    "max_influence": mx,
                    "influence_per_observable": [
                        None if math.isinf(s) else s for s in scores
                    ],
                }
            )
        # unbounded (None) first — they cannot be removed at all — then by
        # descending influence, ties by original position
        rows.sort(
            key=lambda r: (
                r["max_influence"] is not None,
                -(r["max_influence"] or 0.0),
                r["index"],
            )
        )
        return rows[:top] if top else rows

    def _measurement_kind(self, i: int) -> str:
        return (
            "distribution"
            if isinstance(self.observables[i], Distribution)
            else "expectation"
        )

    def _measurement_cost_line(self) -> str | None:
        mc = self.measurement_cost
        if not mc:
            return None
        before, after = mc["terms_before"], mc["terms_after"]
        unit = "term" if before == 1 else "terms"
        unit_a = "term" if after == 1 else "terms"
        note = " (unchanged)" if before == after else ""
        line = f"  measurement cost : {before} {unit} -> {after} {unit_a}{note}"
        if mc.get("distributions"):
            line += " + " + _plural(mc["distributions"], "distribution")
        return line

    def _hardware_impact_lines(self) -> list[str]:
        hw = self.hardware_impact
        if not hw:
            return []
        rates = hw["error_rates"]
        rate_txt = ", ".join(
            f"{k} error {v:.1e}" for k, v in sorted(rates.items(), reverse=True)
        )
        ib = hw["expected_infidelity_before"]
        ia = hw["expected_infidelity_after"]
        fb = hw.get("expected_fidelity_before", 1.0 - ib)
        fa = hw.get("expected_fidelity_after", 1.0 - ia)

        def _factor(txt: str) -> str:
            return f"   (~{txt})" if txt else ""

        def _ratio(r: float) -> str:
            return f"{r:.0f}x" if r < 1e4 else f"{r:.1e}x"

        def _fid(v: float) -> str:
            return f"{v:.2f}" if v >= 0.005 else f"{v:.1e}"

        # infidelity saturates at 1.0 on deep circuits — exactly where the
        # reduction is most dramatic — so switch to fidelity there;
        # infidelity keeps the resolution in the low-error regime
        if max(ib, ia) > 0.9:
            if fb > 0.0 and fa > fb:
                factor = _factor(_ratio(fa / fb) + " better")
            else:  # fidelity_before underflowed: genuinely undefined
                factor = ""
            quantity = f"expected fidelity  {_fid(fb)} -> {_fid(fa)}{factor}"
        else:
            if ia > 0.0 and ib > ia:
                factor = _factor(_ratio(ib / ia) + " lower")
            elif ia == 0.0 and ib > 0.0:
                factor = _factor("all estimated error removed")
            else:
                factor = ""
            quantity = f"expected infidelity  {ib:.1e} -> {ia:.1e}{factor}"
        shift = max(self.certified_bounds, default=0.0)
        return [
            f"  estimated hardware impact ({rate_txt}):",
            f"    {quantity}",
            "    ^ product-of-fidelities ESTIMATE — separate from the",
            "      certified bound below, which is proven:",
            f"    certified ideal-value shift        <= {shift:.1e}",
        ]

    def _empty_circuit_lines(self) -> list[str]:
        """The empty-reduced-circuit verdict: every measurement is
        classically determined and no hardware run is needed.

        Wording rule: "exactly" ONLY when the certified bound is exactly
        zero (the value then holds for the original circuit too). With a
        nonzero bound the value is exact for the REDUCED circuit and the
        bound is what connects it to the original — never say "exactly".
        """
        if self.exact_values is None:
            return []
        lines = []
        for name, value, bound in zip(
            self.observable_names, self.exact_values, self.certified_bounds
        ):
            if isinstance(value, dict):  # distribution on |0...0>
                stmt = (
                    f"outcome {value['outcome']} of {name} has "
                    f"probability {value['probability']:g}"
                )
            else:
                stmt = f"<{name}> = {value:+.6f}"
            if bound == 0.0:
                lines.append(f"  Reduced circuit is empty: {stmt} exactly.")
            else:
                lines.append(
                    f"  Reduced circuit is empty: {stmt}, within "
                    f"{bound:.1e} of the original."
                )
        lines.append("  No quantum execution required.")
        return lines

    def _peak_terms_line(self) -> str:
        """Peak term count shown against the cap it ran under."""
        peak = max(self.max_terms_seen, default=0)
        note = " — cap reached" if any(self.coarsening_events) else ""
        return f"  peak terms : {peak:,} / {self.max_terms:,}{note}"

    def _no_reduction_block(self, indent: str = "  ") -> list[str]:
        """The named-cause, next-command account of a zero-removal run."""
        d = self.diagnosis or {}
        lines = ["No reduction possible for this circuit and observable."]
        why = d.get("why")
        if why:
            lines.extend(
                textwrap.wrap(
                    f"Why: {why}",
                    width=74,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
            )
        suggestions = d.get("suggestions") or []
        if suggestions:
            lines.append(f"{indent}What to try:")
            for s in suggestions:
                cmd, note = s["command"], s["note"]
                if cmd.startswith("--"):
                    lines.append(f"{indent}  {cmd:<20s}  {note}")
                else:
                    lines.append(f"{indent}  {cmd} — {note}")
        expected = d.get("expected_note")
        if expected:
            lines.extend(
                textwrap.wrap(
                    expected,
                    width=74,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
            )
        return lines

    def explain(self) -> str:
        """A plain-language account of what was cut, what was kept, and why."""
        i, o = self.input_metrics, self.output_metrics

        def pct(a: int, b: int) -> str:
            if not a or a == b:
                return "unchanged"
            return f"-{100 * (a - b) / a:.0f}%"

        if self.no_reduction:
            # summary() already carries the full diagnosis (cause, what to
            # try); the CLI prints it right above this, so repeating the
            # block would show a user the same text twice. Lead with the
            # conclusion and add what summary() does not have: the ledger
            # and the error accounting.
            lines = [
                "No reduction possible — the summary names the cause and "
                "what to try; below is the gate-by-gate account.",
            ]
        else:
            # lead with the conclusion, then the statistics (net count:
            # resynthesis both removes and adds gates)
            worst = max(self.certified_bounds, default=0.0)
            lines = [
                f"Removed {i['gates'] - o['gates']} of {i['gates']} gates "
                f"({pct(i['gates'], o['gates'])}); every measured result "
                f"is provably within {worst:.3g} of the original.",
                "",
                "WHAT HAPPENED",
                f"  The circuit shrank from {i['gates']} gates (depth "
                f"{i['depth']}, {i['qubits']} qubits, {i['two_qubit_gates']} "
                f"multi-qubit gates) to {o['gates']} gates (depth {o['depth']}, "
                f"{o['qubits']} qubits, {o['two_qubit_gates']} multi-qubit "
                f"gates): {pct(i['gates'], o['gates'])} gates, "
                f"{pct(i['depth'], o['depth'])} depth, "
                f"{pct(i['qubits'], o['qubits'])} qubits.",
            ]
        by_reason = self.removed_by_reason()
        if by_reason:
            lines.append("")
            lines.append("WHY GATES WERE REMOVED")
            for reason, count in by_reason.items():
                desc = REASON_DESCRIPTIONS.get(reason, reason)
                noun = "gate" if count == 1 else "gates"
                lines.append(f"  {count:>5} {noun}: {desc}.")
        if self.removed_qubits:
            compact = (
                (self.subject or {}).get("options", {}).get("compact", True)
            )
            lines.append("")
            if len(self.removed_qubits) == 1:
                subject = f"Qubit {self.removed_qubits[0]} carries"
                dropped = "was dropped"
                kept = "was kept in place"
                pronoun = "it"
            else:
                subject = f"Qubits {self.removed_qubits} carry"
                dropped = "were dropped"
                kept = "were kept in place"
                pronoun = "them"
            if compact:
                lines.append(
                    f"  {subject} nothing the measurements can see and "
                    f"{dropped}; the report's qubit_map records the "
                    "renumbering."
                )
            else:
                # compact=False keeps every qubit in place: nothing was
                # dropped and no renumbering happened, so say so
                lines.append(
                    f"  {subject} nothing the measurements can see, but "
                    f"{kept} because compact=False; rerun without "
                    f"--no-compact (or with compact=True) to drop "
                    f"{pronoun} and renumber."
                )
        all_rows = self.kept_gate_ledger()
        if all_rows:
            # physical range cap per measurement: an expectation can move by
            # at most 2*||O||, an outcome probability by at most 1. Raw
            # influence bounds can exceed that (true but vacuous); the
            # narrative clamps to the range, the JSON ledger keeps raw.
            caps = [
                1.0
                if isinstance(m, Distribution)
                else 2.0 * m.norm_bound()
                for m in self.observables
            ]
            # every unbounded gate would print the identical line; say it
            # once as a summary instead of once per gate. The full
            # per-gate ledger stays in the JSON report.
            unbounded = [r for r in all_rows if r["max_influence"] is None]
            bounded = [r for r in all_rows if r["max_influence"] is not None]

            def _row_view(row) -> tuple[float, bool]:
                scores = row["influence_per_observable"]
                eff = max(
                    (min(s, c) for s, c in zip(scores, caps) if s is not None),
                    default=0.0,
                )
                clamped = any(
                    s is not None and s >= c
                    for s, c in zip(scores, caps)
                )
                return eff, clamped

            views = [_row_view(r) for r in bounded]
            # a "ranking" in which rows sit at the physical ceiling ranks
            # nothing — the bound saturated, exactly like infidelity on
            # deep circuits. Fold saturated rows into one true statement
            # (>= 2 of them; a single one stays inline as context) and
            # rank only the rows that genuinely differ.
            saturated = [
                (r, v) for r, v in zip(bounded, views) if v[1]
            ]
            ranked = [
                (r, v) for r, v in zip(bounded, views) if not v[1]
            ]
            if len(saturated) < 2:
                ranked = list(zip(bounded, views))
                saturated = []
            max_cap = max(caps, default=0.0)
            lines.append("")
            if ranked:
                lines.append(
                    "WHY THE REST WAS KEPT (most load-bearing first)"
                )
            else:
                lines.append("WHY THE REST WAS KEPT")
            if unbounded:
                lines.append(
                    f"  {_plural(len(unbounded), 'gate')}: no bound "
                    "available (the analysis coarsened at the term cap "
                    "before it could bound "
                    f"{'it' if len(unbounded) == 1 else 'them'})."
                )
            if saturated and not ranked and not unbounded:
                lines.extend(
                    textwrap.wrap(
                        f"None of the "
                        f"{_plural(len(saturated), 'remaining gate')} "
                        "could be bounded below the trivial maximum: "
                        "removing any one of them could shift a measured "
                        f"result by up to {max_cap:.3g}, its entire "
                        "possible range. These influence bounds are "
                        "worst-case over all input states and typically "
                        "sit far above a gate's true effect, so this does "
                        "not mean every gate matters — only that none was "
                        "provably removable.",
                        width=74,
                        initial_indent="  ",
                        subsequent_indent="  ",
                    )
                )
            elif saturated:
                lines.extend(
                    textwrap.wrap(
                        f"{_plural(len(saturated), 'gate')}: no bound "
                        f"below the trivial maximum of {max_cap:.3g} "
                        "(the entire "
                        "range of a measured result). These are "
                        "worst-case bounds over all input states — not "
                        "proof that each gate matters, only that none of "
                        "these was provably removable.",
                        width=74,
                        initial_indent="  ",
                        subsequent_indent="    ",
                    )
                )
            for row, (eff, clamped) in ranked[:5]:
                detail = (
                    "removing it could shift a measured result by up "
                    f"to {eff:.3g}"
                )
                if clamped:
                    detail += " (its entire possible range)"
                lines.append(f"  gate #{row['index']}: {detail}.")
            remaining = len(ranked) - min(len(ranked), 5)
            if remaining > 0:
                lines.append(
                    f"  ... and {remaining} more (full ledger in the JSON "
                    "report)."
                )
        lines.append("")
        lines.append("ERROR ACCOUNTING (proven upper bounds, not estimates)")
        for idx, (name, approx, dust) in enumerate(
            zip(self.observable_names, self.approx_bounds, self.numerical_dust)
        ):
            total = approx + dust
            if self._measurement_kind(idx) == "distribution":
                claim = (
                    f"every outcome probability of {name} is within "
                    f"{total:.3g} of the original"
                )
            else:
                claim = f"<{name}> is within {total:.3g} of the original"
            parts = f"(approximation {approx:.3g} + float noise {dust:.3g})"
            lines.append(f"  {claim} {parts}.")
        if self.epsilon > 0:
            lines.append(
                f"  The requested budget was epsilon = {self.epsilon:g}; the "
                "approximation part above never exceeds it."
            )
        else:
            lines.append(
                "  Exact mode: only rewrites proven to preserve the "
                "measurements were applied; the tiny float-noise figure is "
                "reported instead of being rounded to zero."
            )
        if self.coarsening_events:
            peak = max(self.max_terms_seen, default=0)
            lines.append("")
            lines.append(
                f"  NOTE: on "
                f"{_plural(len(self.coarsening_events), 'occasion')} the "
                "term-level analysis hit its size cap "
                f"({peak:,} / {self.max_terms:,} terms) and fell back to "
                "coarser support tracking — still sound, just less "
                "aggressive. A higher max_terms may reduce further."
            )
        return "\n".join(lines)

    def measurement_block(self) -> dict[str, Any]:
        """Everything a consumer needs to USE the reduced circuit.

        Compaction renumbers qubits, so the observable the caller asked
        for no longer addresses the right wires on the reduced circuit.
        This block carries, per measurement: the original form, the
        REMAPPED form valid on the reduced circuit (canonical string and
        structured terms), and the certified bound — plus the qubit map.
        ``qubit_map_pairs`` ([[old, new], ...], integers) is the
        canonical form; the ``qubit_map`` object is a convenience view
        whose keys are strings only because JSON requires it.
        """
        entries = []
        for i, red in enumerate(self.observables):
            orig = (
                self.original_observables[i]
                if self.original_observables is not None
                else None
            )
            entry: dict[str, Any] = {
                "name": self.observable_names[i],
                "kind": self._measurement_kind(i),
                "original": None if orig is None else str(orig),
                "reduced": str(red),
                "certified_bound": self.certified_bounds[i],
            }
            if isinstance(red, Distribution):
                entry["reduced_qubits"] = list(red.qubits)
            else:
                entry["reduced_terms"] = red.to_dict()["terms"]
            if self.exact_values is not None:
                # empty reduced circuit: the value is classically
                # determined — pipelines can branch on this and skip
                # submitting a job. Together with certified_bound this
                # is the complete claim about the ORIGINAL circuit.
                entry["exact_value"] = self.exact_values[i]
            entries.append(entry)
        return {
            "note": (
                "Measure the 'reduced' observables on the reduced "
                "circuit; they are the caller's observables remapped "
                "through the qubit map. qubit_map_pairs "
                "([[original, reduced], ...]) is the canonical form of "
                "the map; the string-keyed qubit_map object is a "
                "convenience view."
            ),
            "qubit_map_pairs": [
                [int(k), int(v)] for k, v in sorted(self.qubit_map.items())
            ],
            "qubit_map": {
                str(k): v for k, v in sorted(self.qubit_map.items())
            },
            "observables": entries,
        }

    def to_dict(self) -> dict[str, Any]:
        def _num(v: float) -> float | None:
            return None if math.isinf(v) else v

        return {
            # schema_version tracks the report FORMAT, independently of the
            # package version, so consumers can parse across releases
            "schema_version": SCHEMA_VERSION,
            "version": self.version,
            "subject": self.subject,
            "epsilon": self.epsilon,
            "theta": self.theta,
            "observables": self.observable_names,
            "approx_bounds": self.approx_bounds,
            "numerical_dust": self.numerical_dust,
            "certified_bounds": self.certified_bounds,
            "input": self.input_metrics,
            "output": self.output_metrics,
            "removed_gates": [r.to_dict() for r in self.removed_gates],
            "removed_qubits": self.removed_qubits,
            "qubit_map": {str(k): v for k, v in self.qubit_map.items()},
            "measurement": self.measurement_block(),
            "influence_scores": {
                str(k): [_num(x) for x in v]
                for k, v in sorted(self.influence_scores.items())
            },
            "coarsening_events": self.coarsening_events,
            "max_terms_seen": self.max_terms_seen,
            "max_terms": self.max_terms,
            "no_reduction": self.no_reduction,
            "diagnosis": self.diagnosis,
            "measurement_cost": self.measurement_cost,
            "hardware_impact": self.hardware_impact,
            "rounds": self.rounds,
            "timings_ms": self.timings_ms,
            "explanation": {
                "removed_by_reason": self.removed_by_reason(),
                "reason_glossary": {
                    k: REASON_DESCRIPTIONS[k]
                    for k in self.removed_by_reason()
                    if k in REASON_DESCRIPTIONS
                },
                "kept_gates": self.kept_gate_ledger(),
                "measurement_kinds": [
                    self._measurement_kind(i)
                    for i in range(len(self.observable_names))
                ],
                "text": self.explain(),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        i, o = self.input_metrics, self.output_metrics
        if self.no_reduction:
            # a zero-removal run gets a named cause and concrete next
            # commands instead of "shrank by -0%" arithmetic
            lines = self._no_reduction_block()
            lines.append(self._peak_terms_line())
            lines.append(
                f"  rounds: {self.rounds}, "
                f"time: {self.timings_ms.get('total', 0.0):.1f} ms"
            )
            return "\n".join(lines)
        # ordered by what decides whether a hardware run succeeds:
        # two-qubit gates dominate the error budget, depth is what
        # decoherence acts on, active qubits and measurement terms are
        # the resource and shot cost. Total gate count matters least and
        # comes below. The JSON carries everything unchanged.
        lines = [
            f"ACTIVE reduction (epsilon={self.epsilon:g}, "
            f"{_plural(len(self.observable_names), 'observable')})",
        ]
        lines.extend(self._empty_circuit_lines())
        lines += [
            f"  2q+    : {i['two_qubit_gates']:>6} -> {o['two_qubit_gates']:<6}"
            f" (-{i['two_qubit_gates'] - o['two_qubit_gates']})",
            f"  depth  : {i['depth']:>6} -> {o['depth']:<6}"
            f" (-{i['depth'] - o['depth']})",
            f"  qubits : {i['qubits']:>6} -> {o['qubits']:<6}"
            f" (-{i['qubits'] - o['qubits']})",
        ]
        mc_line = self._measurement_cost_line()
        if mc_line:
            lines.append(mc_line)
        for name, bound in zip(self.observable_names, self.certified_bounds):
            lines.append(f"  certified |d<{name}>| <= {bound:.3e}")
        lines.extend(self._hardware_impact_lines())
        lines.append(
            f"  gates  : {i['gates']:>6} -> {o['gates']:<6}"
            f" (-{i['gates'] - o['gates']})"
        )
        by_reason: dict[str, int] = {}
        for r in self.removed_gates:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
        if by_reason:
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_reason.items()))
            lines.append(f"  removed: {parts}")
        lines.append(self._peak_terms_line())
        if any(self.coarsening_events):
            lines.append(
                f"  note: analysis degraded on "
                f"{len(self.coarsening_events)} occasion(s) "
                "(term-count cap); reduction is sound but less aggressive"
            )
        lines.append(
            f"  rounds: {self.rounds}, "
            f"time: {self.timings_ms.get('total', 0.0):.1f} ms"
        )
        return "\n".join(lines)
