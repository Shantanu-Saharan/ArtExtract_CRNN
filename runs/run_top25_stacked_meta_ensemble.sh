#!/bin/bash
#SBATCH --job-name=top25_meta
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/Z3_%j.out
#SBATCH --error=logs/Z3_%j.err

set -euo pipefail

REPO_DIR=$HOME/CRNN
VENV=$HOME/myenv
REPORT=$REPO_DIR/reports/top25_meta_ensemble_report.json
MODELS_OUT=$REPO_DIR/reports/top25_meta_models.pkl

source "$VENV/bin/activate"
cd "$REPO_DIR"
mkdir -p logs reports reports/meta_cache

echo "========================================"
echo "Top-25 stacked meta-ensemble"
echo "Report:     $REPORT"
echo "Meta model: $MODELS_OUT"
echo "GPUs:       2"
echo "========================================"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

python stacked_meta_ensemble.py \
  --train_csv datasets/top25_train_balanced.csv \
  --val_csv datasets/top25_val.csv \
  --root_dir wikiart \
  --batch_size 24 \
  --num_workers 8 \
  --feature_mode logits_probs \
  --tta_views 4 \
  --c_grid 0.03 0.1 0.3 1.0 3.0 10.0 \
  --report_out "$REPORT" \
  --models_out "$MODELS_OUT" \
  --cache_dir reports/meta_cache \
  --model_ckpt checkpoints/linear_probe_top25_siglip2.pt \
  --model_label siglip2_probe \
  --model_image_size 384 \
  --model_use_tta 1 \
  --model_ckpt checkpoints/best_top25_siglip2_ft.pt \
  --model_label siglip2_ft \
  --model_image_size 384 \
  --model_use_tta 1 \
  --model_ckpt checkpoints/best_top25_v2.pt \
  --model_label dino_top25_v2 \
  --model_image_size 448 \
  --model_use_tta 1 \
  --model_ckpt checkpoints/best_top25_a40.pt \
  --model_label dino_top25_a40 \
  --model_image_size 448 \
  --model_use_tta 1 \
  --model_ckpt checkpoints/linear_probe_top25_eva02_1gpu.pt \
  --model_label eva02_probe \
  --model_image_size 448 \
  --model_use_tta 0 \
  --model_ckpt checkpoints/linear_probe_top25_clip_h.pt \
  --model_label clip_h_probe \
  --model_image_size 448 \
  --model_use_tta 0
