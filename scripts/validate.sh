#!/usr/bin/env bash
# The second run strips asserts: soundness checks must be real raises.
set -euo pipefail
cd "$(dirname "$0")/.."
 
python3 -m pytest -q
python3 -O -m pytest -q