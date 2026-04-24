#!/bin/bash
#SBATCH --job-name=top25_v2
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/top25_v2_%j.out
#SBATCH --error=logs/top25_v2_%j.err

# Phase-2 DINO run on L40 GPUs.
# This job resumes from `best_top25_a40.pt` and keeps the higher-resolution
# follow-up that produced `best_top25_v2.pt` for the final ensemble.

REPO_DIR=$HOME/CRNN
VENV=$HOME/myenv
DINO_WEIGHTS=$REPO_DIR/weights/dinov2_vitl14.pth
RESUME_CKPT=$REPO_DIR/checkpoints/best_top25_a40.pt

export LD_LIBRARY_PATH=$HOME/myenv/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $VENV/bin/activate
cd $REPO_DIR
mkdir -p logs checkpoints weights

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NGPU" -lt 1 ]; then NGPU=1; fi
ACCUM=$(( 8 / NGPU ))
if [ "$ACCUM" -lt 1 ]; then ACCUM=1; fi

echo "========================================"
echo "Top-25 Phase 2 - DINOv2 + ArcFace - ${NGPU} L40 GPU(s)"
echo "Resume: best_top25_a40.pt (MeanF1=0.80, Artist=0.881)"
echo "Focus: style texture (448px) + style weight 0.45"
echo "Effective batch: $((16 * NGPU * ACCUM))"
echo "========================================"

if [ ! -f "$DINO_WEIGHTS" ]; then
    echo "ERROR: DINOv2 weights not found: $DINO_WEIGHTS"; exit 1
fi

if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: $RESUME_CKPT not found — run the A40 job first"; exit 1
fi

echo "Resuming from: $RESUME_CKPT"

torchrun --nproc_per_node=$NGPU --master_port=29760 train_multitask_ddp.py \
    --train_csv datasets/top25_train.csv \
    --val_csv   datasets/top25_val.csv \
    --root_dir  wikiart \
    \
    --num_artist_classes 25 \
    --num_style_classes  27 \
    --num_genre_classes  7 \
    --use_hybrid \
    \
    --backbone        dinov2_vitl14 \
    --use_cross_attn \
    --pretrained_path $DINO_WEIGHTS \
    --resume_checkpoint $RESUME_CKPT \
    \
    --batch_size   16 \
    --accum_steps  $ACCUM \
    --epochs       100 \
    --image_size   448 \
    --num_workers  8 \
    \
    --lr                 5e-5 \
    --backbone_lr_scale  0.08 \
    --weight_decay       0.05 \
    --use_llrd \
    --llrd_decay         0.80 \
    \
    --freeze_epochs       0 \
    --no_sgdr_restart \
    --sgdr_t0             100 \
    \
    --use_ema \
    --ema_decay           0.9995 \
    \
    --use_tta \
    --tta_views           12 \
    \
    --artist_weight       0.40 \
    --style_weight        0.45 \
    --genre_weight        0.15 \
    \
    --use_arcface \
    --arcface_s           32.0 \
    --arcface_m_artist    0.30 \
    --arcface_m_style     0.30 \
    --arcface_tasks       "artist,style" \
    \
    --artist_focal_gamma  1.5 \
    --style_focal_gamma   3.0 \
    --genre_focal_gamma   2.0 \
    \
    --artist_label_smoothing  0.05 \
    --style_label_smoothing   0.05 \
    --genre_label_smoothing   0.02 \
    \
    --artist_logit_tau    0.3 \
    --style_logit_tau     0.0 \
    --genre_logit_tau     0.0 \
    \
    --class_weight_power  0.75 \
    \
    --augmentation_prob   0.20 \
    --aug_end_prob        0.05 \
    --grad_clip           0.5 \
    --early_stop_patience 15 \
    \
    --use_bfloat16 \
    --log_per_class_f1 \
    \
    --save_path    checkpoints/best_top25_v2.pt \
    --curves_path  checkpoints/curves_top25_v2.png \
    --history_json checkpoints/history_top25_v2.json

echo "Done! Checkpoint: checkpoints/best_top25_v2.pt"
