#!/bin/bash
#SBATCH --job-name=top25_clip_h_probe
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/clip_h_probe_%j.out
#SBATCH --error=logs/clip_h_probe_%j.err

REPO_DIR=$HOME/CRNN
VENV=$HOME/myenv
WEIGHTS=$REPO_DIR/weights/clip_vith14.pth
SAVE=$REPO_DIR/checkpoints/linear_probe_top25_clip_h.pt

source $VENV/bin/activate
cd $REPO_DIR
mkdir -p logs checkpoints weights

echo "========================================"
echo "Top-25 CLIP ViT-H/14 linear probe"
echo "Weights: $WEIGHTS"
echo "Save:    $SAVE"
echo "========================================"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: weights not found: $WEIGHTS"
    echo "Download with: python download_pretrained.py --backbone clip_vith14"
    exit 1
fi

python linear_probe.py \
    --backbone clip_vith14 \
    --pretrained_path "$WEIGHTS" \
    --train_csv datasets/top25_train_balanced.csv \
    --val_csv datasets/top25_val.csv \
    --root_dir wikiart \
    --num_artist_classes 25 \
    --num_style_classes 27 \
    --num_genre_classes 7 \
    --image_size 448 \
    --batch_size 12 \
    --epochs 25 \
    --lr 8e-4 \
    --weight_decay 0.02 \
    --num_workers 8 \
    --save_path "$SAVE"
