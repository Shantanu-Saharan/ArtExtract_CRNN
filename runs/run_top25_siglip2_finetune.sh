#!/bin/bash
#SBATCH --job-name=top25_sg2ft
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/siglip2_ft_%j.out
#SBATCH --error=logs/siglip2_ft_%j.err

set -euo pipefail

REPO_DIR=$HOME/CRNN
VENV=$HOME/myenv
MODEL_DIR=$REPO_DIR/weights/siglip2-so400m-patch16-384
PROBE_CKPT=$REPO_DIR/checkpoints/linear_probe_top25_siglip2.pt
SAVE=$REPO_DIR/checkpoints/best_top25_siglip2_ft.pt
HISTORY=$REPO_DIR/reports/top25_siglip2_ft_history.json

source "$VENV/bin/activate"
cd "$REPO_DIR"
mkdir -p logs checkpoints reports weights/hf_cache

echo "========================================"
echo "Top-25 SigLIP 2 full fine-tune"
echo "Model dir:        $MODEL_DIR"
echo "Probe checkpoint: $PROBE_CKPT"
echo "Save path:        $SAVE"
echo "GPUs requested:   $SLURM_NTASKS_PER_NODE"
echo "========================================"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "SigLIP 2 checkpoint not found locally."
  exit 1
fi

EXTRA_ARGS=()
if [ -f "$SAVE" ]; then
  echo "[INFO] Found existing fine-tune checkpoint, resuming from $SAVE"
  EXTRA_ARGS+=(--resume_checkpoint "$SAVE")
elif [ -f "$PROBE_CKPT" ]; then
  echo "[INFO] Warm-starting heads from probe checkpoint $PROBE_CKPT"
  EXTRA_ARGS+=(--probe_checkpoint "$PROBE_CKPT")
else
  echo "[INFO] No probe checkpoint found, starting from pretrained SigLIP 2 only"
fi

echo "starting siglip2 finetune..."
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
  --epochs 18 \
  --warmup_epochs 1.0 \
  --freeze_backbone_epochs 1 \
  --head_lr 8e-5 \
  --backbone_lr 1.5e-5 \
  --weight_decay 0.02 \
  --min_lr_ratio 0.10 \
  --grad_clip 1.0 \
  --num_workers 4 \
  --artist_weight 0.35 \
  --style_weight 0.41 \
  --genre_weight 0.24 \
  --artist_focal_gamma 1.0 \
  --style_focal_gamma 1.25 \
  --genre_focal_gamma 1.0 \
  --artist_label_smoothing 0.03 \
  --style_label_smoothing 0.05 \
  --genre_label_smoothing 0.02 \
  --llrd_decay 0.92 \
  --use_llrd \
  --use_gradient_checkpointing \
  --use_tta \
  --tta_views 2 \
  --use_bfloat16 \
  --early_stop_patience 5 \
  --save_path "$SAVE" \
  --history_json "$HISTORY" \
  "${EXTRA_ARGS[@]}"
