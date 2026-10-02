#!/usr/bin/env bash
# Demo of ADE at alpha = 1/16 (Table 1) on one model:
#   1) ADE_-L (compression only)
#   2) ADE (with light fold-in adaptation)
# Each run is evaluated on the model's benchmark with the compressed delta applied by ADE's kernels.
#   bash run.sh [model]      model in {magicoder-7b (default), wizardmath-7b, wizardmath-13b,
#                                      wizardcoder-13b, llava-7b, llava-13b}
# Environment: ADE_MODEL_ROOT (local model directories; default ./models, missing models are
# downloaded), ADE_CACHE_DIR (stage-1 cache; default ./cache/stage1), GPU (default 0).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:-magicoder-7b}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

python src/main.py --config configs/${MODEL}_untuned.yaml --output_dir outputs/${MODEL}_untuned
python src/main.py --config configs/${MODEL}.yaml --output_dir outputs/${MODEL}

cat outputs/${MODEL}_untuned/eval/summary.json outputs/${MODEL}/eval/summary.json
