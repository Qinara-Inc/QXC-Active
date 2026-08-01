# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Template for future entries:

## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Deprecated
- ...

### Removed
- ...

### Fixed
- ...

### Security
- ...
-->

## [Unreleased]

## [0.3.0] - 2026-08-03

First public open-source release of ACTIVE (`qxc-active`), a primitive of the
QXC platform, developed by Qinara. Released under the MIT license.

### Added

- Observable-aware circuit reduction (`active_reduce`) with certified
  per-observable error bounds: exact mode preserves all expectation values;
  approximate mode (`epsilon > 0`) prunes further under a proven budget.
- Per-observable batch mode (`--per-observable`, `qxc_active.batch`) producing
  one maximally-reduced bundle per observable.
- Native OpenQASM 2.0 and 3.0 import and OpenQASM 3.0 export, with no external
  toolchain.
- Self-verifying output bundles (`--out-dir`) with a SHA-256 manifest, checked
  by `active verify-bundle`.
- Clifford prefix resynthesis (exact, self-checking, on by default).
- Independent brute-force verifier (`qxc_active.verifier`) and dense
  statevector simulator (`qxc_active.sim`) used by the test suite.

### Changed

- The observable tracker uses a hybrid representation: a dict below a
  term-count threshold and parallel numpy arrays above it. Results across the
  promotion boundary are equal up to floating-point summation order; removal
  decisions are unaffected.
- The earlier per-qubit term index was removed after profiling showed it cost
  more to maintain than the scans it saved. Reduced circuits and removal lists
  are byte-identical with and without it.

Versions before 0.3.0 were internal development milestones and were never
published.
