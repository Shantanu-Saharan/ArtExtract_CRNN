#!/bin/bash
# SigLIP2 rescue run focused on recovering style while keeping the stronger
# pooled-style warm start intact.

#SBATCH --job-name=top25_sg2r1
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0-18:00:00
#SBATCH --output=logs/siglip2_rescue_%j.out
#SBATCH --error=logs/siglip2_rescue_%j.err

set -euo pipefail

REPO_DIR=$HOME/CRNN
VENV=$HOME/myenv
MODEL_DIR=$REPO_DIR/weights/siglip2-so400m-patch16-384
BASE_CKPT=$REPO_DIR/checkpoints/best_top25_siglip2_ft.pt
SAVE=$REPO_DIR/checkpoints/best_top25_siglip2_rescue.pt
HISTORY=$REPO_DIR/reports/top25_siglip2_rescue_history.json

source "$VENV/bin/activate"
cd "$REPO_DIR"
mkdir -p logs checkpoints reports weights/hf_cache

echo "========================================"
echo "Top-25 SigLIP2 rescue run"
echo "Model dir:      $MODEL_DIR"
echo "Init checkpoint:$BASE_CKPT"
echo "Save path:      $SAVE"
echo "========================================"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

EXTRA_ARGS=()
if [ -f "$SAVE" ]; then
  echo "[INFO] Resuming existing rescue run from $SAVE"
  EXTRA_ARGS+=(--resume_checkpoint "$SAVE")
elif [ -f "$BASE_CKPT" ]; then
  echo "[INFO] Warm-starting from best SigLIP2 checkpoint $BASE_CKPT"
  EXTRA_ARGS+=(--init_checkpoint "$BASE_CKPT")
else
  echo "[INFO] Base checkpoint missing, starting from scratch"
fi

torchrun --nproc_per_node="$SLURM_NTASKS_PER_NODE" train_siglip2_multitask_ddp.py \
  --model_name_or_path "$MODEL_DIR" \
  --cache_dir weights/hf_cache \
  --train_csv datasets/top25_train_balanced.csv \
  --val_csv datasets/top25_val.csv \
  --root_dir wikiart \
  --num_artist_classes 25 \
  --num_style_classes 27 \
  --num_genre_classes 7 \
  --image_size 384 \
  --batch_size 4 \
  --accum_steps 8 \
  --epochs 16 \
  --warmup_epochs 1.0 \
  --freeze_backbone_epochs 1 \
  --head_lr 5e-5 \
  --backbone_lr 8e-6 \
  --weight_decay 0.02 \
  --min_lr_ratio 0.10 \
  --grad_clip 1.0 \
  --num_workers 4 \
  --artist_weight 0.33 \
  --style_weight 0.44 \
  --genre_weight 0.23 \
  --artist_focal_gamma 1.0 \
  --style_focal_gamma 1.3 \
  --genre_focal_gamma 1.0 \
  --artist_label_smoothing 0.03 \
  --style_label_smoothing 0.05 \
  --genre_label_smoothing 0.02 \
  --llrd_decay 0.92 \
  --use_llrd \
  --use_style_fusion \
  --use_gradient_checkpointing \
  --use_tta \
  --tta_views 2 \
  --use_bfloat16 \
  --early_stop_patience 5 \
  --save_path "$SAVE" \
  --history_json "$HISTORY" \
  "${EXTRA_ARGS[@]}"
