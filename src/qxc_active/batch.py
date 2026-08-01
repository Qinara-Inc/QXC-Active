# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Per-observable batch reduction: one maximally-reduced circuit per
observable, tied together by an index.

The joint mode (the default) produces one circuit valid for every
observable — a gate survives if ANY observable needs it. For a
Hamiltonian of many terms measured against one ansatz that is the wrong
trade: a term that reduces to nothing on its own gets dragged back to
full size by its neighbours. This module runs each observable
independently — each reduction is identical to running that observable
alone — and reports where the measurement budget should go.

Orchestration only: no back-propagation work is shared between
observables (that is a real optimisation and a separate design problem).
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import __version__
from .bundle import write_bundle
from .errors import ActiveError
from .ir import Circuit
from .paulis import Distribution
from .reduce import active_reduce
from .report import SCHEMA_VERSION, _plural

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _dir_names(observables: list) -> list[str]:
    """Filesystem-safe, unique directory name per observable.

    Named observables (e.g. Distributions) use a sanitised form of
    their name; unnamed ones get positional obs0, obs1, ...
    """
    names: list[str] = []
    used: set[str] = set()
    for i, spec in enumerate(observables):
        label = getattr(spec, "name", None)
        if label is None and isinstance(spec, Distribution):
            label = str(spec)
        base = _SAFE.sub("_", label).strip("_")[:40] if label else f"obs{i}"
        base = base or f"obs{i}"
        name = base
        n = 2
        while name in used:
            name = f"{base}__{n}"
            n += 1
        used.add(name)
        names.append(name)
    return names


def _label(spec) -> str:
    """Human label for progress lines and the index."""
    return str(spec)


def _run_one(payload: tuple) -> dict:
    """Reduce ONE observable and write its bundle. Module-level so a
    process pool can pickle it; also called directly in serial mode.

    Never raises: a failing observable is a row with an "error", so one
    bad term cannot abort a 500-term batch.
    """
    i, circuit, spec, out_dir, dir_name, options = payload
    row: dict = {"index": i, "label": _label(spec), "dir": dir_name}
    t0 = time.perf_counter()
    try:
        reduced, report = active_reduce(circuit, [spec], **options)
        if out_dir is not None:
            write_bundle(Path(out_dir) / dir_name, reduced, report)
        entry = report.measurement_block()["observables"][0]
        row.update(
            {
                "gates": report.output_metrics["gates"],
                "two_qubit_gates": report.output_metrics["two_qubit_gates"],
                "depth": report.output_metrics["depth"],
                "qubits": report.output_metrics["qubits"],
                "certified_bound": report.certified_bounds[0],
                "no_reduction": report.no_reduction,
                "coarsened": bool(report.coarsening_events),
                "error": None,
            }
        )
        if "exact_value" in entry:
            row["exact_value"] = entry["exact_value"]
    except ActiveError as exc:
        row.update({"error": str(exc)})
    row["wall_ms"] = (time.perf_counter() - t0) * 1e3
    return row


def reduce_per_observable(
    circuit: Circuit,
    observables: list,
    out_dir: str | Path | None,
    *,
    jobs: int = 1,
    progress=None,
    **options,
) -> dict:
    """Run one independent reduction per observable.

    Each reduction is byte-identical to running that observable alone;
    results are collected in input order regardless of ``jobs``. Returns
    the index structure (also written to ``out_dir/index.json`` with
    ``summary.md`` and a top-level manifest when ``out_dir`` is given).

    ``progress`` is called with one line per finished observable (order
    of completion, which under ``jobs > 1`` may differ from input
    order — the artifacts never do).
    """
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    dir_names = _dir_names(observables)
    in_2q = circuit.two_qubit_count()
    payloads = [
        (i, circuit, spec, None if out_dir is None else str(out_dir),
         dir_names[i], options)
        for i, spec in enumerate(observables)
    ]

    total = len(payloads)

    def _announce(row: dict) -> None:
        if progress is None:
            return
        if row["error"] is not None:
            body = f"ERROR: {row['error'].splitlines()[0][:60]}"
        else:
            body = f"2q {in_2q} -> {row['two_qubit_gates']}"
        progress(
            f"[{row['index'] + 1:>4}/{total}] {row['label'][:28]:<28} "
            f"{body}  ({row['wall_ms'] / 1e3:.1f}s)"
        )

    rows: list[dict | None] = [None] * total
    if jobs <= 1:
        for p in payloads:
            row = _run_one(p)
            rows[row["index"]] = row
            _announce(row)
    else:
        # observables are genuinely independent; completion order is
        # nondeterministic but rows are keyed by input index, so every
        # artifact and listing is deterministic regardless of jobs
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            from concurrent.futures import as_completed

            futures = [ex.submit(_run_one, p) for p in payloads]
            for fut in as_completed(futures):
                row = fut.result()
                rows[row["index"]] = row
                _announce(row)

    # the joint baseline: what a single shared circuit would cost. This
    # is the comparison the mode exists to expose, and it cannot be
    # known without actually running the joint reduction once.
    joint = None
    ok_specs = [
        observables[r["index"]] for r in rows if r and r["error"] is None
    ]
    if len(ok_specs) >= 1:
        t0 = time.perf_counter()
        try:
            joint_reduced, _joint_rep = active_reduce(
                circuit, ok_specs, **options
            )
            joint = {
                "two_qubit_gates": joint_reduced.two_qubit_count(),
                "gates": joint_reduced.n_gates,
                "wall_ms": (time.perf_counter() - t0) * 1e3,
            }
            if progress is not None:
                progress(
                    f"[joint baseline] one shared circuit: "
                    f"2q {in_2q} -> {joint['two_qubit_gates']}  "
                    f"({joint['wall_ms'] / 1e3:.1f}s)"
                )
        except ActiveError:
            joint = None  # comparison unavailable; the rows still stand

    ok = [r for r in rows if r["error"] is None]
    per_total_2q = sum(r["two_qubit_gates"] for r in ok)
    index = {
        "schema_version": SCHEMA_VERSION,
        "tool": "qxc-active",
        "version": __version__,
        "mode": "per-observable",
        "input": {
            "qubits": circuit.n_qubits,
            "gates": circuit.n_gates,
            "two_qubit_gates": in_2q,
        },
        "observables": rows,
        "totals": {
            "n_observables": total,
            "n_failed": sum(1 for r in rows if r["error"] is not None),
            "n_no_execution": sum(
                1 for r in ok if r["gates"] == 0
            ),
            "n_coarsened": sum(1 for r in ok if r["coarsened"]),
            # each term's circuit runs once per term
            "two_qubit_executions": per_total_2q,
            # the shared circuit would run once PER TERM
            "joint_two_qubit_executions": (
                None if joint is None
                else joint["two_qubit_gates"] * len(ok)
            ),
            "joint_two_qubit_gates": (
                None if joint is None else joint["two_qubit_gates"]
            ),
        },
    }
    if out_dir is not None:
        _write_index_files(Path(out_dir), index, dir_names, rows)
    return index


def _write_index_files(
    out: Path, index: dict, dir_names: list[str], rows: list[dict]
) -> None:
    import hashlib

    def _write(name: str, text: str) -> None:
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    _write("index.json", json.dumps(index, indent=2) + "\n")
    _write("summary.md", render_summary_md(index))
    bundle_dirs = sorted(
        r["dir"] for r in rows if r["error"] is None
    )

    def _sha(name: str) -> str:
        h = hashlib.sha256()
        h.update((out / name).read_bytes())
        return h.hexdigest()

    manifest = {
        "tool": "qxc-active",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "mode": "per-observable",
        "files": {n: _sha(n) for n in ("index.json", "summary.md")},
        # nested bundles carry their own manifests; verify-bundle
        # recurses through this list
        "bundles": bundle_dirs,
    }
    _write("manifest.json", json.dumps(manifest, indent=2) + "\n")


def render_summary_md(index: dict) -> str:
    """The decision document: free terms first, then cost-sorted."""
    rows = index["observables"]
    ok = [r for r in rows if r["error"] is None]
    failed = [r for r in rows if r["error"] is not None]
    free = [r for r in ok if r["gates"] == 0]
    costly = sorted(
        (r for r in ok if r["gates"] > 0),
        key=lambda r: (-r["two_qubit_gates"], -r["gates"], r["index"]),
    )
    t = index["totals"]
    inp = index["input"]

    lines = [
        "# Per-observable reduction",
        "",
        f"Input: {inp['qubits']} qubits, {inp['gates']} gates "
        f"({inp['two_qubit_gates']} two-qubit). "
        f"{_plural(t['n_observables'], 'observable')}, "
        f"{_plural(t['n_failed'], 'failure')}.",
        "",
    ]
    if free:
        lines += [
            "## No quantum execution required",
            "",
            "These terms reduced to nothing: their value on |0...0> is "
            "classically determined (shown with its certified bound). "
            "They cost no hardware at all.",
            "",
            "| observable | exact value | certified bound |",
            "|---|---|---|",
        ]
        for r in free:
            v = r.get("exact_value")
            v_txt = (
                f"{v:+.6f}" if isinstance(v, (int, float))
                else json.dumps(v)
            )
            lines.append(
                f"| `{r['label']}` | {v_txt} | {r['certified_bound']:.3e} |"
            )
        lines.append("")
    if costly:
        lines += [
            "## Remaining cost, most expensive first",
            "",
            "| observable | 2q gates | gates | depth | qubits | "
            "certified bound | coarsened | bundle |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in costly:
            lines.append(
                f"| `{r['label']}` | {r['two_qubit_gates']} | "
                f"{r['gates']} | {r['depth']} | {r['qubits']} | "
                f"{r['certified_bound']:.3e} | "
                f"{'yes' if r['coarsened'] else 'no'} | `{r['dir']}/` |"
            )
        lines.append("")
    if failed:
        lines += ["## Failed", ""]
        for r in failed:
            first = r["error"].splitlines()[0]
            lines.append(f"* `{r['label']}` — {first}")
        lines.append("")

    lines += ["## Total quantum work", ""]
    per = t["two_qubit_executions"]
    lines.append(
        f"* **This mode:** {_plural(per, 'two-qubit execution')} "
        f"(each term's own circuit, run once per term)."
    )
    if t["joint_two_qubit_executions"] is not None:
        joint = t["joint_two_qubit_executions"]
        lines.append(
            f"* **Joint mode:** {_plural(joint, 'two-qubit execution')} "
            f"(one shared {t['joint_two_qubit_gates']}-2q-gate circuit, "
            f"run once per term)."
        )
        if joint > 0:
            saved = 100.0 * (joint - per) / joint
            n_free = t["n_no_execution"]
            verb = "needs" if n_free == 1 else "need"
            lines.append(
                f"* **Saving: {saved:.0f}%** of two-qubit work"
                + (
                    f"; {_plural(n_free, 'term')} {verb} no hardware "
                    "at all."
                    if n_free
                    else "."
                )
            )
    if t["n_coarsened"]:
        lines.append(
            f"* {_plural(t['n_coarsened'], 'term')} hit the term cap "
            "(coarsened analysis) — a higher --max-terms may reduce "
            "those further."
        )
    return "\n".join(lines) + "\n"
