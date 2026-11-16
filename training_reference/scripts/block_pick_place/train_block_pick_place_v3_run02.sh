#!/usr/bin/env bash
set -euo pipefail
set +u
source /home/lixin/miniconda3/etc/profile.d/conda.sh
conda activate tools
set -u
PROJECT=/home/lixin/act_gripper
MODE="${1:-train}"
case "$MODE" in
  train) STEPS=30000; NAME=block_pick_place_v3_act_run02 ;;
  smoke) STEPS=100; NAME=block_pick_place_v3_run02_smoke100 ;;
  *) exit 2 ;;
esac
OUTPUT="${2:-${PROJECT}/experiments/${NAME}}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT"; exit 2; }
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT}/lerobot/src:${PROJECT}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT}/.cache/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd "$PROJECT"
python -u pretrain_act_bc.py --config "$PROJECT/configs/train_block_pick_place_v3_run02.json" \
  --data_dir "$PROJECT/dataset/block_pick_place_v3_no_ep6" --output_dir "$OUTPUT" \
  --steps "$STEPS" --batch_size 16 --num_workers 4 --device cuda
