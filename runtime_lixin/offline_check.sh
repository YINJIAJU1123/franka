#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec python -u "$BASELINE_ROOT/offline_checks.py"
