# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.3.x | Yes |
| < 0.3 | No |

## Reporting a vulnerability

Please report suspected vulnerabilities privately — do not open a public
issue. Either:

- email: info@qinara.co.uk
- use the repository's private vulnerability reporting
  ("Security" → "Report a vulnerability" on GitHub), once the repository is
  public.

You will receive an acknowledgement, and a fix or advisory will be
coordinated with you before public disclosure.

## Scope notes

ACTIVE is a local, offline library and CLI. It has **no network access and no
telemetry**: it reads circuit files you pass it, computes, and writes results
to the paths you specify. The most security-relevant surface is parsing of
untrusted OpenQASM input; parser crashes or resource-exhaustion issues on
crafted inputs are in scope and welcome as reports.
