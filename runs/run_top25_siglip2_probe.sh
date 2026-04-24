#!/bin/bash
#SBATCH --job-name=top25_sg2
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/siglip2_probe_%j.out
#SBATCH --error=logs/siglip2_probe_%j.err

set -euo pipefail

REPO_DIR=$HOME/CRNN
VENV=$HOME/myenv
MODEL_DIR=$REPO_DIR/weights/siglip2-so400m-patch16-384
SAVE=$REPO_DIR/checkpoints/linear_probe_top25_siglip2.pt

source "$VENV/bin/activate"
cd "$REPO_DIR"
mkdir -p logs checkpoints weights/hf_cache

echo "========================================"
echo "Top-25 SigLIP 2 linear probe (1 GPU)"
echo "Model dir: $MODEL_DIR"
echo "Save:      $SAVE"
echo "========================================"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "SigLIP 2 checkpoint not found locally, downloading..."
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_dir = os.path.expanduser("~/CRNN/weights/siglip2-so400m-patch16-384")
snapshot_download(
    repo_id="google/siglip2-so400m-patch16-384",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)
print(f"[OK] Downloaded SigLIP 2 to {model_dir}")
PY
fi

python siglip2_linear_probe.py \
  --model_name_or_path "$MODEL_DIR" \
  --cache_dir weights/hf_cache \
  --train_csv datasets/top25_train_balanced.csv \
  --val_csv datasets/top25_val.csv \
  --root_dir wikiart \
  --num_artist_classes 25 \
  --num_style_classes 27 \
  --num_genre_classes 7 \
  --batch_size 8 \
  --epochs 15 \
  --lr 8e-4 \
  --weight_decay 0.02 \
  --num_workers 8 \
  --save_path "$SAVE"
