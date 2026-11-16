#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo 'This entry takes no arguments and always disables motion.' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec python -u "$BASELINE_ROOT/original_runner.py" \
  --act_ckpt_dir "$BASELINE_ROOT/checkpoint" --act_ckpt_file act_policy_ema.pth \
  --config "$BASELINE_ROOT/runtime_config.json" --urdf "$BASELINE_ROOT/fr3.urdf" \
  --device cuda --hz 30 --allow-motion 0 --dry-run-steps 300 \
  --auto-reset 0 --gripper-reset hold \
  --gripper-close-threshold .0693 --gripper-open-threshold .074 \
  --head-topic /camera/cam_head/color/image_raw --wrist-topic /camera/cam_wrist/color/image_raw \
  --max-linear-velocity .12 --max-angular-velocity .35 \
  --log-dir "$BASELINE_ROOT/reports/observe_only"
