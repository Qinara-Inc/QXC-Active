# qxc-active

**Observable-aware quantum circuit reduction with provable error bounds.**

ACTIVE strips a circuit down to the parts that affect the observables you
actually measure, removing gates, interactions and whole qubits that provably
cannot change your measured outputs. In *exact* mode every expectation value
is preserved. In *approximate* mode it prunes further under an error budget
you set, and returns a proven upper bound on how much each observable could
have shifted.

Runtime dependencies are numpy and an OpenQASM parser. No compiler, no vendor
SDK.

## Install

```
pip install -e .                 # core library and CLI
pip install -e .[test]           # + test tooling
pip install -e .[bench]          # + Qiskit, for the independent cross-check tests
```

## Running it

The shortest form. If the circuit declares measurements, ACTIVE uses them:

```
active reduce circuit.qasm
```

```
No observable given. Using the 16 qubits this circuit measures:
  Distribution([0, 1, ..., 15])
Override with --observables "Z0*Z1" or --sample-qubits 0,1,2.
```

Usually you want to say what you are measuring:

```
active reduce circuit.qasm --observables "Z0*Z1"
active reduce circuit.qasm --observables "Z0" "X1*Y2" "0.5*Z0*Z1 + 0.25*X2"
active reduce circuit.qasm --observables-file hamiltonian.txt
active reduce circuit.qasm --sample-qubits 0,1,2
```

Add `--out-dir` to get the full artifact bundle rather than a printed summary:

```
active reduce circuit.qasm --observables "Z0*Z1" --out-dir out/
```

From Python:

```python
from qxc_active import load_qasm, active_reduce, dumps_qasm

circ = load_qasm("circuit.qasm")            # OpenQASM 2.0 or 3.0, auto-detected
reduced, report = active_reduce(circ, ["Z0*Z1"])
print(report.summary())
open("reduced.qasm", "w").write(dumps_qasm(reduced))
```

### Options

| flag | effect |
|---|---|
| `--out-dir DIR` | write the full bundle instead of just printing |
| `--explain` | prose account of what was cut, what was kept, and why |
| `--epsilon E` | approximate pruning within a certified budget (experimental, see *Known limitations*) |
| `--per-observable` | one maximally-reduced circuit per observable (see below) |
| `--jobs N` | parallelism for `--per-observable`; results identical for any N |
| `--no-compact` | keep the original qubit numbering |
| `--max-terms N` | raise the analysis cap when a circuit coarsens |
| `--error-rates "1q=1e-4,2q=5e-3"` | add an estimated hardware-impact block |
| `--json` | machine-readable output on stdout |

## Choosing an observable

The observable is **the quantity you want the answer to**, not a property of
the circuit. ACTIVE needs it because it is what makes a gate removable: a gate
that cannot change your measured value can be deleted. Different observables
therefore give different reductions from the same circuit.

**Expectation values.** Energies, correlators, magnetisation. Write the Pauli
operator you are estimating:

```
--observables "Z0"                          single-site Z
--observables "Z0*Z1"                       two-body correlator
--observables "X1*Y2*Z4"                    a product of Paulis
--observables "0.5*Z0*Z1 + 0.25*X2 - Y0"    a weighted sum, e.g. a Hamiltonian term
--observables "IZXY"                        dense string, qubit 0 leftmost
```

**Measurement outcomes.** Sampling bitstrings, as in Grover, Shor, or any
algorithm read out in the computational basis. Use the distribution form,
which preserves the full joint distribution over those qubits:

```
--sample-qubits 0,1,2
```

or in Python, `Distribution([0, 1, 2])`.

**Many terms at once.** A Hamiltonian, One per line in a file, `#` for
comments:

```
# NH3 STO-3G, selected terms
0.31*Z0*Z1
-0.12*X0*X1*Y2*Y3
Z7
```

```
active reduce ansatz.qasm --observables-file hamiltonian.txt --out-dir out/
```

**What is accepted:** any Hermitian real-weighted Pauli sum, and `Distribution`
over a set of qubits. Complex coefficients are rejected (an observable must be
Hermitian), as are raw matrices and non-Pauli operators; decompose to Paulis
first. A Pauli on a qubit the circuit does not have is an error.

**On passing several observables.** One run with several observables produces a
*single* circuit valid for all of them: a gate survives if *any* observable
needs it. That is right when you intend to submit one job. For many Hamiltonian
terms you almost certainly want `--per-observable`.

## Per-observable mode

Measuring a Hamiltonian means one measurement setting per term or commuting
group, so sharing a single circuit across all terms keeps gates most of them do
not need. `--per-observable` reduces each observable independently and writes
one bundle for each:

```
active reduce ansatz.qasm --observables-file hamiltonian.txt \
    --per-observable --jobs 4 --out-dir out/
```

```
out/
  index.json        one row per observable: label, bundle dir, gate counts,
                    certified bound, exact value, wall time, error
  summary.md        the same as a readable table, plus totals
  manifest.json     SHA-256 over the tree; active verify-bundle out/ checks it
  obs0/             a complete bundle per observable
  obs1/
  ...
```

`summary.md` leads with terms that reduced to nothing, which are classically
determined and need no hardware at all, then lists the rest by remaining cost,
and reports total two-qubit executions against what a single joint reduction
would have cost.

A failing observable is recorded as a row in `index.json` and does not abort the
batch. Exit codes: `0` all succeeded, `1` every term reduced nothing, `2` any
term rejected.

## Understanding the output

### The summary

```
ACTIVE reduction (epsilon=0, 1 observable)
  2q+    :   3318 -> 42     (-3276)
  depth  :   5123 -> 109    (-5014)
  qubits :     16 -> 5      (-11)
  measurement cost : 1 term -> 1 term (unchanged)
  certified |d<obs0>| <= 4.354e-14
  gates  :   1024 -> 147    (-877)
  removed: outside_support: 812, invariant: 61, cancelled: 4
  peak terms : 11 / 20,000
  rounds: 2
```

| line | meaning |
|---|---|
| `2q+` | two-qubit-or-larger gates. The number that matters most: these dominate hardware error |
| `depth` | circuit depth, what decoherence acts on |
| `qubits` | qubits still in use. Fewer means less routing overhead on real hardware |
| `measurement cost` | Pauli terms you must measure. ACTIVE never increases this |
| `certified` | **proven** upper bound on how far the measured value can have moved |
| `removed` | which pass removed what: `outside_support` never touches the observable, `invariant` commutes with it, `cancelled` annihilates against a partner |
| `peak terms` | how close the analysis came to its cap. At the cap it coarsens and reduces less |

When a circuit reduces to nothing the answer is classically determined and
reported directly:

```
Reduced circuit is empty: <obs0> = +1.000000 exactly.
No quantum execution required.
```

The `exact_value` field appears in the JSON too, so a pipeline can skip
submitting the job.

### The bundle

`--out-dir` writes six files:

| file | contents |
|---|---|
| `reduced.qasm` | the reduced circuit, OpenQASM 3 |
| `observables.json` | the observables remapped to the reduced circuit's qubit numbering, plus the qubit map |
| `report.json` | every metric, machine-readable, with a `subject` fingerprint of the input |
| `certificate.json` | the rewrite ledger: each step, the bound it charged, the running total |
| `report.md` | the summary and explanation in readable form |
| `manifest.json` | SHA-256 of every other file; check with `active verify-bundle out/` |

**`observables.json` is not optional.** Compaction renumbers qubits, so a
reduced circuit without its qubit map cannot be used correctly. Measure the
observable given there, on the circuit given there.

### What to do with it

Compile the reduced circuit exactly as you would have compiled the original,
then measure the observable from `observables.json`. ACTIVE is a pre-pass, not
a compiler: running it first and compiling second compounds, because the
compiler works on a smaller circuit and its own optimisations still apply.

### Exit codes

| code | meaning |
|---|---|
| 0 | reduction succeeded |
| 1 | nothing removed. A valid result, not an error; CI can branch on it |
| 2 | input rejected (control flow, opaque, reset, parse failure) or invalid usage |
| 3 | internal invariant violated (`BudgetViolationError`). Should be unreachable; if it fires, do not trust the reduction |

---

## How it works

**The measurement runs backward, not the circuit forward.** A lightcone pass
keeps every gate that could eventually touch a measured qubit. ACTIVE instead
drags each observable backward through the circuit in the Heisenberg picture.
At every step the observable is a real-weighted sum of Pauli strings, updated
exactly through each gate via its Pauli transfer expansion. Walking from the
last gate to the first:

- a gate touching none of the qubits the observable currently occupies is
  deleted;
- a gate that leaves the observable exactly invariant, meaning it commutes with
  what the observable has become at that point, is also deleted. This is the
  step a lightcone cannot take: a CZ buried deep inside the cone still dies if
  the back-propagated observable is Z-type there;
- Clifford gates map each Pauli term to exactly one term, so support can
  *shrink*. Z⊗Z pulled through a CX becomes a single-qubit Z, and everything
  upstream on the other qubit dies with it.

**Three further exact passes run to fixpoint,** since each removal can unlock
others:

1. **Cancellation and merging.** Inverse pairs annihilate across commuting
   gates between them, rotations of the same family merge, near-identity gates
   vanish.
2. **Initial-state simplification.** Qubits start in |0⟩, so gates acting
   trivially on the reachable state are removed: a controlled gate whose
   control is provably still |0⟩, a phase gate on an unentangled |0⟩.
3. **Qubit elimination.** Qubits with no remaining gates and no observable
   support are dropped, with an old-to-new index map recorded in the report.

Termination is guaranteed: every pass either strictly shrinks the circuit or
changes nothing.

**Where the certified bound comes from.** Two facts do the work: conjugation
by the rest of the circuit never changes an operator's norm, and
|⟨0…0|Q|0…0⟩| ≤ ‖Q‖. So if a rewrite changes the back-propagated observable
by an operator of norm δ at any point, the final expectation value changes by
at most δ. The **influence** of a gate is exactly that δ for its hypothetical
removal, computed on the gate's own one to three qubits as a small exact
matrix calculation. Bounds of successive rewrites add by the triangle
inequality. In approximate mode gates are removed cheapest-influence-first
while the per-observable running total stays within ε, and the report states
the bound actually accumulated. Whole-unitary rewrites such as cancellations
are bounded by |Δ⟨O⟩| ≤ 2‖O‖·‖ΔU‖ with ‖O‖ ≤ Σ|coefficients|.

**Float accounting is explicit.** IEEE-754 arithmetic is not exact, so even in
exact mode the report carries a separate `numerical_dust` bound covering every
discarded floating-point crumb, rather than claiming an error of zero that
cannot be backed up. The total `certified_bounds = approx_bounds +
numerical_dust` is a true upper bound on each
|⟨O⟩_original − ⟨O⟩_reduced|, brute-force verified against an independent
simulator across thousands of random circuits in the test suite.

**Degradation is sound, not silent.** Non-Clifford rotations split Pauli terms
in two, so rotation-dense circuits grow the term count exponentially. That is
fundamental, and it is *why* such circuits are hard to simulate, rather than a
shortcut. The term count is capped (`max_terms`, default 20,000). At the cap
the analysis either coarsens to support-only tracking for that observable, or
in approximate mode truncates the smallest terms and charges them against ε.
Reduction quality degrades; correctness does not.

## Guarantees

Each row is enforced by a test, and the suite runs a second time under
`python -O` so that no guarantee depends on an `assert`.

| Guarantee | Test |
|---|---|
| Exact mode preserves all expectation values (≤1e-8 including simulator noise) | `test_exact.py`, `test_property.py` |
| The certified bound is a true upper bound on the actual error | `test_approx.py`, `test_property.py` (brute force) |
| Approximate charges never exceed ε | `test_approx.py`, `test_property.py` |
| Deterministic and reproducible across runs and hash seeds | `test_passes.py`, `test_property.py` |
| Idempotent (a true fixpoint) | `test_passes.py`, `test_property.py` |
| Arbitrary gates including raw-matrix unitaries | `test_exact.py` |
| Sound degradation under term-cap pressure | `test_exact.py`, `test_approx.py` |
| Per-observable bundles identical to separate runs | `test_per_observable.py` |
| Clear errors on invalid input, no crash on valid input | `test_edge.py`, `test_qasm.py` |
| Bounds verified against an independent implementation | `test_crosscheck_qiskit.py` |

## Known limitations

Stated here rather than left to be discovered.

**Approximate mode (`epsilon > 0`) is experimental and rarely worth its cost.**
A gate's influence is bounded worst-case over all input states, and removals
accumulate by triangle inequality, so the certified bound typically exceeds the
true error by one to two orders of magnitude. It prunes little beyond what exact
mode already achieves, at extra runtime cost, so exact mode is the recommended
default.

**Term count grows exponentially on rotation-dense circuits.** Fundamental, as above. The analysis coarsens or truncates rather than becoming wrong.

**The analysis has a tail.** Local observables stay cheap even at high qubit
counts: cost scales with the number of Pauli terms the observable carries, not
the qubit count. Wide observables on rotation-dense circuits are the slow path,
because the term count can grow large there. A plain lightcone pass is uniformly
cheaper, so on large circuits weigh the analysis cost against the compilation
and execution it saves.

**Some workloads see little or no benefit.** Highly entangling, rotation-dense
ansätze such as QAOA and hardware-efficient VQE reduce least, and with a global
observable may not reduce at all. Deliberately structureless
circuits such as Quantum Volume give modest reductions. The largest gains come
from Fourier-based algorithms, Hamiltonian simulation, and arithmetic or
Clifford-dominated circuits.

**ACTIVE is not a compiler.** It deletes gates; it does not resynthesise
unitary blocks. A resynthesising optimiser will beat it on raw gate count for
resynthesis-heavy circuits. Run ACTIVE first, then compile as normal. The two
compound.

**Unitary circuits only.** Classical control flow, `reset` and `opaque` are
rejected by design; see *Scope* below.

## Complexity and performance

- Time O(gates × terms × rounds), where *terms* is the number of observable
  Pauli terms. For Clifford circuits this often shrinks; otherwise it grows
  toward `max_terms`.
- Memory O(terms × qubits / 64), since Pauli strings are stored as bit-mask pairs.
- Approximate mode multiplies runtime by the threshold-ladder size.
- The tracker uses a hybrid representation: a dict below a term-count
  threshold and numpy arrays above it, since neither wins at both scales.
  Profiling also showed the earlier per-qubit term index cost far more to
  maintain than the scans it saved, so it was removed in 0.3.0. Reduced
  circuits and removal lists are byte-identical either way.

## API

```python
active_reduce(
    circuit: Circuit,
    observables,            # str | PauliTerm | Observable | Distribution | sequence
    epsilon: float = 0.0,   # per-observable error budget; 0 = exact
    *,
    max_terms: int = 20_000,
    cancellation_lookback: int = 100,
    max_rounds: int | None = None,
    dust_tolerance: float = 1e-12,
    compact: bool = True,   # drop removed qubits and renumber
    verify: bool = False,   # opt-in brute-force check (small circuits only)
    clifford_resynthesis: bool = True,
) -> tuple[Circuit, Report]
```

**Observable syntax:** `"Z0"`, `"X1*Z4"`, dense `"IZXY"` (qubit 0 leftmost),
weighted sums `"0.5*Z0*Z1 + 0.25*X2 - Y0"`. `Distribution([0, 1, 2])`
preserves the full joint measurement distribution on those qubits; with ε > 0
the certified bound applies to each individual outcome probability. This is
deliberately a per-outcome bound, not a total-variation claim, which would
carry a 2^|M| factor.

Per-observable batches:

```python
from qxc_active.batch import reduce_per_observable

index = reduce_per_observable(circuit, observables, out_dir="out/", jobs=4)
for row in index["observables"]:
    print(row["label"], row["dir"], row["two_qubit_gates"], row["certified_bound"])
```

**`Report` fields:** `certified_bounds` (per observable, equal to
`approx_bounds` + `numerical_dust`), `removed_gates` (index, reason and cost
each), `removed_qubits`, `qubit_map`, `observables` (remapped),
`influence_scores`, `input_metrics` / `output_metrics`, `measurement_cost`,
`coarsening_events`, `rounds`, `timings_ms`, `epsilon`, `theta`, plus
`summary()`, `explain()` and `to_json()`.

**Empty results state their answer.** When a circuit reduces to nothing, the
expectation value is determined classically and reported directly, with an
`exact_value` field in the JSON so a pipeline can skip submitting a job:

```
Reduced circuit is empty: <obs0> = +1.000000 exactly.
No quantum execution required.
```

**`Circuit.measured_qubits`** records the qubits a loaded program measures, in
program order. Informational only: it takes no part in reduction and does not
affect circuit equality or the report fingerprint. Circuits built in code have
an empty tuple.

**Clifford resynthesis** (default on): a circuit prefix composed only of
Clifford gates produces a stabilizer state. ACTIVE tracks it on an integer
tableau and replaces the prefix with a freshly synthesized preparation of the
identical state whenever that is strictly smaller. Exact, self-checking (it falls back to the original on any surprise) and free
on non-Clifford circuits.

**Explain-your-work:** `report.explain()` renders a plain-language account of
what was cut and why, what was kept and its influence, and the full error
accounting. The same content is in `report.to_dict()["explanation"]`, and
`active reduce ... --explain` prints it.

**Independent verifier** (`qxc_active.verifier`): a deliberately simple
brute-force checker used by the test suite and available via `verify=True`.
Never imported on the default path. It checks one concrete reduction by brute
force and is not a formal equivalence checker.

## OpenQASM support

ACTIVE reads **OpenQASM 2.0 and 3.0 natively**, with no external toolchain and
no conversion step:

```python
circ = qxc_active.load_qasm("circuit.qasm")   # dialect auto-detected
```

The dialect is taken from the `OPENQASM` header; pass
`dialect="qasm2" | "qasm3"` for files with a missing or incorrect one. A 2.0
file and its 3.0 equivalent produce an identical circuit and reduction, proven
by tests: 2.0 input goes through a strict, line-preserving source transform
(header, `qelib1.inc`, `qreg`/`creg`, `measure a -> b;`) into the same parser,
so a gate is never silently rewritten. The full `qelib1.inc` gate set is
covered; gates absent from `stdgates.inc` (`u0`, `rccx`, `rc3x`, `c3x`,
`c3sqrtx`, `c4x`) are supplied as their canonical qelib1 definitions, each
verified against an independent implementation exactly, including global phase.

API: `load_qasm(path)`, `loads_qasm(text)`, `dump_qasm(circ, path)`,
`dumps_qasm(circ)`, `measured_qubits(text)`, `convert_qasm2_to_qasm3(text)`.
Custom gate definitions are inlined and register broadcast is supported.
Export is OpenQASM 3.0.

### Scope

ACTIVE's certified bounds are stated for a fixed sequence of gates, which
means a unitary circuit. Classical control flow (`if (c == 1) ...`) has no
fixed gate sequence, `reset` is not unitary, and an `opaque` gate has no
defined unitary, so all three are rejected, with an error naming the
construct and its source line, never a raw parser message. Measurements and
barriers are dropped silently, since they do not affect observable reduction;
a file's measured qubits are available via `measured_qubits(text)` and on
`Circuit.measured_qubits`.

ACTIVE reports a certified bound for its own prunings. It does not perform
formal equivalence verification of circuits.

## CLI

```
active reduce IN.qasm --observables ... [--epsilon E] [--out-dir D]
                      [--max-terms N] [--error-rates SPEC] [--no-compact]
                      [--print-qasm] [--json] [--explain]
                      [--per-observable [--jobs N]]
active verify-bundle DIR
active info IN.qasm
```

Flags are described under *Running it*; exit codes under *Understanding the
output*. `active info` prints a circuit's metrics without reducing it, and
`active verify-bundle` recomputes every hash in an output directory, including
nested per-observable bundles.

## Development

```
python -m venv .venv
.venv/bin/pip install -e .[test] -c constraints.txt      # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest                               # 823 tests
.venv/bin/python -m pytest -q && python -O -m pytest -q  # both modes
./scripts/validate.sh                                    # full gate
```

Pinned versions for development and CI are in `constraints.txt`. Versioning is
semantic, and `qxc_active.__version__` is the single source of truth, and
`pyproject.toml` reads it from there. Logging goes through the standard
`logging` module under the `qxc_active` logger: DEBUG for per-round progress,
INFO for degradation events. No network access, no telemetry, no hidden state.

## Dependencies and licenses

| Package | Scope | License |
|---|---|---|
| numpy | core | BSD-3-Clause |
| openqasm3[parser] | core (QASM I/O) | Apache-2.0 |
| antlr4-python3-runtime | core (via openqasm3 parser) | BSD-3-Clause |
| pytest | test only | MIT |
| hypothesis | test only, never imported by core | MPL-2.0 (weak file-level copyleft) |
| qiskit | cross-check tests only, never imported by core | Apache-2.0 |
| qiskit-qasm3-import | cross-check tests only (OpenQASM 3 loading on qiskit 2.x) | Apache-2.0 |

The core library imports only numpy and, lazily for I/O, openqasm3.

## Credits and citation

ACTIVE is a primitive of the **QXC** platform, developed by [Qinara](https://qinara.co.uk) and authored by
Swapnil Deshmukh (<s.deshmukh@qinara.co.uk>, Qinara).

Copyright (c) 2026 Qinara. Released under the [MIT license](LICENSE).

If you use ACTIVE in academic work, please cite it using the metadata in
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button uses it
directly).