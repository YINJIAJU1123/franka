#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash run_real.sh [check|run] (default: check)' >&2
  exit 2
fi
MODE="${1:-check}"
case "$MODE" in
  check|run) ;;
  *) echo 'Usage: bash run_real.sh [check|run]' >&2; exit 2 ;;
esac
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec python -u "$BASELINE_ROOT/real_launch.py" "$MODE"
