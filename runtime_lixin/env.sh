#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /home/lixin/miniconda3/etc/profile.d/conda.sh
conda activate tools
source /opt/ros/humble/setup.bash
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BASELINE_ROOT:$BASELINE_ROOT/lerobot/src:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1 N_ACTION_STEPS=1 ACT_WRIST_RELIABILITY=reliable
unset RMW_IMPLEMENTATION FASTDDS_BUILTIN_TRANSPORTS RMW_FASTRTPS_USE_SHM
cd "$BASELINE_ROOT"
