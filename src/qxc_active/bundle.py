# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""The output bundle: one directory containing everything needed to USE
a reduction, self-verifying via a hash manifest.

A reduced circuit on its own is unsafe: compaction renumbers qubits, so
the observable the caller asked for no longer addresses the right wires.
The bundle therefore always pairs ``reduced.qasm`` with
``observables.json`` (the remapped observables and the qubit map) — the
latter is never optional.

Files:

* ``reduced.qasm``       — the reduced circuit, OpenQASM 3
* ``observables.json``   — the measurement block, standalone
* ``report.json``        — the full report (timings live in the manifest)
* ``certificate.json``   — the rewrite ledger with running bound totals
* ``report.md``          — human summary + explain prose
* ``manifest.json``      — tool/schema versions, resolved options,
  timestamp, per-run timings, and a SHA-256 of every other file

Determinism: the same input and options produce byte-identical files;
everything non-deterministic (timestamp, timings) lives in
``manifest.json`` only, so the rest of the bundle can be hash-compared.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .errors import ActiveError
from .ir import Circuit
from .qasm import dumps_qasm
from .report import SCHEMA_VERSION, Report

MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _certificate(report: Report) -> dict:
    """The rewrite ledger: every removal step with its charged bound and
    the running total, ending at the final certified bound."""
    steps = []
    running = [0.0] * len(report.observable_names)
    for r in report.removed_gates:  # ordered by original gate index
        running = [a + b for a, b in zip(running, r.cost_per_observable)]
        steps.append(
            {
                "index": r.index,
                "kind": r.reason,
                "gate": {"name": r.name, "qubits": list(r.qubits)},
                "bound_charged": list(r.cost_per_observable),
                "running_total": list(running),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "observables": list(report.observable_names),
        "steps": steps,
        "removal_totals": running,
        "approx_bounds": list(report.approx_bounds),
        "numerical_dust": list(report.numerical_dust),
        "certified_bounds": list(report.certified_bounds),
        "note": (
            "certified_bounds = approx_bounds + numerical_dust. Numerical "
            "dust also accrues on kept-gate conjugations, so the final "
            "bound can exceed the ledger's running removal total."
        ),
    }


def write_bundle(out_dir: str | Path, reduced: Circuit, report: Report) -> Path:
    """Write the full output bundle into ``out_dir`` (created if needed)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    def _write(name: str, text: str) -> None:
        with open(d / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

    _write("reduced.qasm", dumps_qasm(reduced))
    _write(
        "observables.json",
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "measurement": report.measurement_block(),
            },
            indent=2,
        )
        + "\n",
    )
    # timings are per-run noise; they live in the manifest ONLY, so
    # every other file is byte-identical for identical input and options
    timings = dict(report.timings_ms)
    neutral = dataclasses.replace(report, timings_ms={})
    _write("report.json", json.dumps(neutral.to_dict(), indent=2) + "\n")
    _write(
        "certificate.json",
        json.dumps(_certificate(report), indent=2) + "\n",
    )
    _write(
        "report.md",
        "# ACTIVE reduction report\n\n## Summary\n\n```\n"
        + neutral.summary()
        + "\n```\n\n## Explanation\n\n```\n"
        + neutral.explain()
        + "\n```\n\n(Run timings live in manifest.json.)\n",
    )
    files = [
        "reduced.qasm",
        "observables.json",
        "report.json",
        "certificate.json",
        "report.md",
    ]
    manifest = {
        "tool": "qxc-active",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "options": (report.subject or {}).get("options", {}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "timings_ms": timings,
        "files": {name: _sha256(d / name) for name in files},
    }
    _write(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
    return d


def verify_bundle(out_dir: str | Path) -> list[str]:
    """Recompute every hash in the manifest; return a list of problems
    (empty = bundle verifies).

    A per-observable index manifest lists nested bundle directories
    under ``bundles``; those are verified recursively, so one call
    covers the whole tree.
    """
    d = Path(out_dir)
    manifest_path = d / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ActiveError(f"no {MANIFEST_NAME} in {d} — not a bundle")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = manifest["files"]
    except (ValueError, KeyError) as exc:
        raise ActiveError(f"malformed {MANIFEST_NAME} in {d}: {exc}") from exc
    problems = []
    for name, expected in sorted(listed.items()):
        p = d / name
        if not p.is_file():
            problems.append(f"{name}: MISSING")
            continue
        actual = _sha256(p)
        if actual != expected:
            problems.append(
                f"{name}: MISMATCH (manifest {expected[:16]}..., "
                f"actual {actual[:16]}...)"
            )
    for sub in manifest.get("bundles", ()):
        subdir = d / sub
        if not (subdir / MANIFEST_NAME).is_file():
            problems.append(f"{sub}/: MISSING nested bundle")
            continue
        try:
            problems.extend(
                f"{sub}/{p}" for p in verify_bundle(subdir)
            )
        except ActiveError as exc:
            problems.append(f"{sub}/: {exc}")
    return problems
