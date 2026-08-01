# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Command-line interface: ``active reduce``, ``active info`` and
``active verify-bundle``.

Examples::

    active reduce bell.qasm --observables Z0 "0.5*Z0*Z1 + 0.5*X0"
    active reduce big.qasm --observables Z0 --epsilon 0.01 --out-dir out/
    active verify-bundle out/
    active info big.qasm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .bundle import verify_bundle, write_bundle
from .errors import ActiveError, BudgetViolationError
from .qasm import dumps_qasm, load_qasm
from .reduce import active_reduce
from .report import circuit_metrics

_EXIT_CODES_HELP = """\
exit codes:
  0  reduction succeeded
  1  nothing removed (a valid result, not an error — CI can branch on it)
  2  input rejected (control flow, opaque, reset, parse failure) or
     invalid usage
  3  internal invariant violated (BudgetViolationError) — should be
     unreachable; if it fires, do not trust the reduction
"""

_OUTPUT_REMOVED = """\
--output was removed in 0.3.0. A reduced circuit on its own is unsafe:
compaction renumbers qubits, so the observable you asked for no longer
addresses the qubits you think it does.
Use --out-dir DIR instead. The bundle contains reduced.qasm together with
observables.json, which carries the remapped observable and the qubit map."""

_REPORT_REMOVED = """\
--report was removed in 0.3.0.
Use --out-dir DIR instead; the bundle's DIR/report.json is the same full
report, alongside observables.json (the remapped observable and qubit map,
without which the reduced circuit cannot be used correctly)."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="active",
        description=(
            "ACTIVE: reduce a quantum circuit to only what affects the "
            "measured observables, with certified error bounds."
        ),
        epilog=_EXIT_CODES_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"qxc-active {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    red = sub.add_parser(
        "reduce", help="reduce a circuit with respect to observables"
    )
    red.add_argument("input", help="input OpenQASM 3 file")
    red.add_argument(
        "--observables",
        "--observable",  # explicit singular alias; without it argparse
        # treats --observable as an ambiguous prefix of --observables and
        # --observables-file and errors out
        "-o",
        nargs="+",
        default=[],
        metavar="OBS",
        help=(
            'observables, e.g. Z0, "X1*Z4", IZZ (dense, qubit 0 leftmost) '
            'or sums like "0.5*Z0*Z1 + 0.25*X2"'
        ),
    )
    red.add_argument(
        "--observables-file",
        metavar="FILE",
        help="file with one observable per line (# comments allowed)",
    )
    red.add_argument(
        "--sample-qubits",
        action="append",
        default=[],
        metavar="Q,Q,...",
        help=(
            "preserve the full measurement distribution on these qubits "
            '(e.g. "0,1,2"); may be given multiple times'
        ),
    )
    red.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=0.0,
        help="per-observable error budget (default 0 = exact mode)",
    )
    red.add_argument(
        "--out-dir",
        metavar="DIR",
        help=(
            "write the output bundle here: reduced.qasm, observables.json "
            "(remapped observables + qubit map — required to use the "
            "circuit), report.json, certificate.json, report.md, "
            "manifest.json (SHA-256 of every file)"
        ),
    )
    # removed flags: fail loudly and helpfully, never silently do nothing
    red.add_argument("--output", metavar="FILE", help=argparse.SUPPRESS)
    red.add_argument("--report", metavar="FILE", help=argparse.SUPPRESS)
    red.add_argument(
        "--max-terms",
        type=int,
        default=20_000,
        help="Pauli-term cap per observable before sound degradation",
    )
    red.add_argument(
        "--per-observable",
        action="store_true",
        help=(
            "reduce each observable independently: one maximally-reduced "
            "bundle per observable under --out-dir (required), plus "
            "index.json and summary.md. Exit 0 when all succeeded, 1 "
            "when every term reduced to nothing removed, 2 if any term "
            "was rejected"
        ),
    )
    red.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "process-pool parallelism for --per-observable (default 1: "
            "serial); results and ordering are identical for any N"
        ),
    )
    red.add_argument(
        "--error-rates",
        metavar="SPEC",
        help=(
            'per-gate error rates, e.g. "1q=1e-4,2q=5e-3"; enables the '
            "ESTIMATED hardware-impact block (product-of-fidelities "
            "model, separate from the certified bound)"
        ),
    )
    red.add_argument(
        "--no-compact",
        action="store_true",
        help="keep original qubit numbering (do not drop removed qubits)",
    )
    red.add_argument(
        "--print-qasm",
        action="store_true",
        help="print the reduced circuit QASM to stdout",
    )
    red.add_argument(
        "--json",
        action="store_true",
        help="print the JSON report to stdout instead of the text summary",
    )
    red.add_argument(
        "--explain",
        action="store_true",
        help="print a plain-language account of what was cut/kept and why",
    )

    info = sub.add_parser("info", help="show circuit metrics")
    info.add_argument("input", help="input OpenQASM file (2.0 or 3.0)")

    verify = sub.add_parser(
        "verify-bundle",
        help="recompute every hash in a bundle's manifest.json and "
        "report mismatches",
    )
    verify.add_argument("bundle", metavar="DIR", help="bundle directory")
    return parser


_NO_OBSERVABLE_ERROR = """\
no observable given, and this circuit declares no measurements.

An observable is the quantity you want the answer to. It tells ACTIVE which
gates can be removed, so different observables give different reductions.

  --observables "Z0"                  expectation of Z on qubit 0
  --observables "0.5*Z0*Z1 + 0.25*X2" a weighted sum, e.g. a Hamiltonian term
  --sample-qubits 0,1,2               the joint measurement distribution"""


def _qubit_list(qubits: tuple[int, ...]) -> str:
    qs = sorted(qubits)
    if len(qs) <= 8:
        return "[" + ", ".join(map(str, qs)) + "]"
    return f"[{qs[0]}, {qs[1]}, ..., {qs[-1]}]"


def _default_observable(circuit):
    """Distribution over the circuit's own declared measurements, with
    the announcement that makes the default explicit. A silent default
    would be worse than the old error; an announced one that names the
    override is not."""
    from .paulis import Distribution

    n = len(circuit.measured_qubits)
    noun = "qubit" if n == 1 else "qubits"
    announcement = (
        f"No observable given. Using the {n} {noun} this circuit "
        "measures:\n"
        f"  Distribution({_qubit_list(circuit.measured_qubits)})\n"
        'Override with --observables "Z0*Z1" or --sample-qubits 0,1,2.'
    )
    return Distribution(circuit.measured_qubits), announcement


def _parse_error_rates(spec: str) -> dict:
    """Parse ``"1q=1e-4,2q=5e-3"`` into ``{"1q": 1e-4, "2q": 5e-3}``."""
    out: dict = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ActiveError(
                f"--error-rates expects KEY=VALUE pairs, got {part!r} "
                '(e.g. "1q=1e-4,2q=5e-3")'
            )
        try:
            out[key.strip()] = float(value)
        except ValueError as exc:
            raise ActiveError(
                f"--error-rates value {value!r} is not a number"
            ) from exc
    if not out:
        raise ActiveError("--error-rates is empty")
    return out


def _load_observables(args: argparse.Namespace) -> list:
    obs: list = list(args.observables)
    if args.observables_file:
        try:
            with open(args.observables_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        obs.append(line)
        except OSError as exc:
            raise ActiveError(
                f"cannot read observables file {args.observables_file!r}: {exc}"
            ) from exc
    for spec in getattr(args, "sample_qubits", []):
        from .paulis import Distribution

        try:
            qubits = [int(t) for t in spec.replace(" ", "").split(",") if t]
        except ValueError as exc:
            raise ActiveError(
                f"--sample-qubits expects comma-separated integers, got {spec!r}"
            ) from exc
        obs.append(Distribution(qubits))
    # an empty list is not an error here: main() falls back to the
    # circuit's own declared measurements, or explains what is needed
    return obs


def _run_per_observable(args, circuit, observables, announcement,
                        error_rates) -> int:
    """One independent reduction per observable (see --per-observable).

    Exit codes: 0 all observables succeeded; 1 every observable ran but
    reduced nothing (a valid result — the joint rule, applied per
    batch); 2 if any observable was rejected. Errors never abort the
    batch: the failing row is recorded in index.json.
    """
    from .batch import reduce_per_observable

    if not args.out_dir:
        raise ActiveError(
            "--per-observable writes one bundle per observable and "
            "needs --out-dir DIR"
        )
    if args.jobs < 1:
        raise ActiveError("--jobs must be >= 1")
    if announcement:
        print(announcement, file=sys.stderr if args.json else sys.stdout)
    progress = None
    if not args.json:
        # a 500-term run must not be silent; stderr keeps stdout clean
        def progress(line: str) -> None:  # noqa: E306
            print(line, file=sys.stderr)

    index = reduce_per_observable(
        circuit,
        observables,
        args.out_dir,
        jobs=args.jobs,
        progress=progress,
        epsilon=args.epsilon,
        max_terms=args.max_terms,
        compact=not args.no_compact,
        error_rates=error_rates,
    )
    if args.json:
        print(json.dumps(index, indent=2))
    else:
        print((Path(args.out_dir) / "summary.md").read_text(
            encoding="utf-8"
        ))
    totals = index["totals"]
    if totals["n_failed"]:
        return 2
    rows = index["observables"]
    if rows and all(r["no_reduction"] for r in rows):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Exit codes: 0 = reduced successfully; 1 = ran fine but nothing was
    removed (a valid result, so pipelines can branch on it); 2 = input
    rejected or invalid usage; 3 = internal invariant violated
    (BudgetViolationError) — do not trust the reduction.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            circuit = load_qasm(args.input)
            print(json.dumps(circuit_metrics(circuit), indent=2))
            return 0

        if args.command == "verify-bundle":
            problems = verify_bundle(args.bundle)
            if problems:
                for p in problems:
                    print(p, file=sys.stderr)
                noun = "problem" if len(problems) == 1 else "problems"
                print(
                    f"bundle verification FAILED "
                    f"({len(problems)} {noun})",
                    file=sys.stderr,
                )
                return 2
            print(f"bundle verified: every hash in {args.bundle} matches")
            return 0

        if args.output is not None:
            print(_OUTPUT_REMOVED, file=sys.stderr)
            return 2
        if args.report is not None:
            print(_REPORT_REMOVED, file=sys.stderr)
            return 2

        circuit = load_qasm(args.input)
        observables = _load_observables(args)
        announcement = None
        if not observables:
            if circuit.measured_qubits:
                default, announcement = _default_observable(circuit)
                observables = [default]
            else:
                raise ActiveError(_NO_OBSERVABLE_ERROR)
        error_rates = (
            _parse_error_rates(args.error_rates) if args.error_rates else None
        )
        if args.per_observable:
            return _run_per_observable(args, circuit, observables,
                                       announcement, error_rates)
        reduced, report = active_reduce(
            circuit,
            observables,
            epsilon=args.epsilon,
            max_terms=args.max_terms,
            compact=not args.no_compact,
            error_rates=error_rates,
        )
        # stdout may be a pipe whose reader has gone away (| head, less,
        # grep -m1 — ordinary terminal use). A broken pipe must not
        # produce a traceback, and — crucially — must not eat the exit
        # code: only the output emission is guarded, and control falls
        # through to the normal return below.
        pipe_broken = False

        def _emit(*print_args, **print_kw) -> None:
            nonlocal pipe_broken
            if pipe_broken:
                return
            try:
                print(*print_args, **print_kw)
            except BrokenPipeError:
                pipe_broken = True

        if args.json:
            # the announcement must not corrupt piped JSON, but a silent
            # default is unacceptable: route it to stderr in json mode
            if announcement:
                print(announcement, file=sys.stderr)
            _emit(report.to_json())
        else:
            if announcement:
                _emit(announcement)
            _emit(report.summary())
        if args.explain and not args.json:
            _emit()
            _emit(report.explain())
        if args.out_dir:
            # file output is independent of stdout health
            write_bundle(args.out_dir, reduced, report)
            if not args.json:
                _emit(f"  bundle written to {args.out_dir}")
        if args.print_qasm:
            _emit(dumps_qasm(reduced), end="")
            if any(k != v for k, v in report.qubit_map.items()):
                # inspection is legitimate; silently handing someone a
                # circuit whose qubits moved is not (stderr keeps pipes
                # clean)
                print(
                    "warning: compaction renumbered qubits — the printed "
                    "circuit's qubit indices do not match the input. Use "
                    "--out-dir DIR to get reduced.qasm together with "
                    "observables.json, which carries the qubit map and "
                    "the remapped observables.",
                    file=sys.stderr,
                )
        if pipe_broken:
            # catching the exception is not enough: the interpreter
            # flushes sys.stdout at shutdown, which would re-raise and
            # print "Exception ignored in: <_io.TextIOWrapper ...>" to
            # stderr. Point the underlying fd at devnull so the final
            # flush has somewhere harmless to go.
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
            except OSError:
                pass
        # exit 1 when nothing was removed: the run itself succeeded, but
        # pipelines want to branch on "did this actually help"
        return 1 if report.no_reduction else 0
    except BudgetViolationError as exc:
        # should be unreachable; if it fires, the reduction is untrusted
        print(f"INTERNAL INVARIANT VIOLATED: {exc}", file=sys.stderr)
        return 3
    except ActiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
