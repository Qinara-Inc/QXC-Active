# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Qinara

"""Exception types raised by qxc_active.

All errors raised on invalid user input derive from :class:`ActiveError`,
so callers can catch a single type. Internal invariant violations that are
load-bearing for soundness raise dedicated :class:`ActiveError` subclasses
(e.g. :class:`BudgetViolationError`) via unconditional checks; only
non-load-bearing documentation invariants use plain ``assert``.
"""

from __future__ import annotations


class ActiveError(Exception):
    """Base class for all qxc_active errors."""


class InvalidCircuitError(ActiveError):
    """The circuit IR is malformed (bad qubit indices, non-unitary matrix, ...)."""


class InvalidObservableError(ActiveError):
    """An observable specification could not be parsed or is malformed."""


class UnsupportedGateError(ActiveError):
    """A gate name is not in the gate registry and no matrix was supplied."""


class QasmImportError(ActiveError):
    """The OpenQASM 3 input is invalid or uses an unsupported construct."""


class QasmExportError(ActiveError):
    """The circuit cannot be expressed in the supported OpenQASM 3 subset."""


class InvalidOptionError(ActiveError):
    """A reduction option (epsilon, max_terms, ...) has an invalid value."""


class BudgetViolationError(ActiveError):
    """The accumulated error charges exceeded the requested budget.

    This is a bug, never bad input — if raised, the reduction must not be
    trusted. It is raised by an unconditional runtime check (not an
    ``assert``), so the guarantee survives ``python -O``.
    """
