#!/bin/bash
# Plan A Pipeline: NR-Rec-FUS Pose Estimator -> neural-ex INR Image Reconstruction
# Usage: bash bridge_nr_rec_fus/run_pipeline.sh [--skip_train] [--skip_inr] [--gpu 0]

set -e

GPU=0
SKIP_TRAIN=false
SKIP_INR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --skip_train) SKIP_TRAIN=true; shift ;;
        --skip_inr) SKIP_INR=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."

echo "============================================"
echo "Plan A Pipeline"
echo "============================================"
echo "GPU: $GPU"
echo "Skip training: $SKIP_TRAIN"
echo "Skip INR eval: $SKIP_INR"
echo ""

# Step 1: Prepare data
echo "=== Step 1: Prepare uvfdata2 data for NR-Rec-FUS ==="
python bridge_nr_rec_fus/step1_prepare_data.py \
    --images_dir ./MyMoE/uvfdata2 \
    --pose_file ./MyMoE/coords3d/uvfdata2.xlsx \
    --output_h5 ./bridge_nr_rec_fus/data/uvfdata2_res4.h5 \
    --resample_factor 4 \
    --num_samples 30 \
    --window_stride 5

# Step 2: Train pose estimator
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "=== Step 2: Train NR-Rec-FUS Pose Estimator ==="
    python bridge_nr_rec_fus/step2_train_pose_estimator.py \
        --data_dir ./bridge_nr_rec_fus/data \
        --h5_file uvfdata2_res4.h5 \
        --num_samples 30 \
        --batch_size 4 \
        --lr 1e-4 \
        --num_epochs 500 \
        --gpu $GPU \
        --save_dir ./bridge_nr_rec_fus/models
else
    echo ""
    echo "=== Step 2: SKIPPED (using existing model) ==="
fi

# Step 3: Predict poses
echo ""
echo "=== Step 3: Predict poses with trained model ==="
python bridge_nr_rec_fus/step3_predict_poses.py \
    --images_dir ./MyMoE/uvfdata2 \
    --model_path ./bridge_nr_rec_fus/models/best_pose_estimator.pth \
    --output_excel ./bridge_nr_rec_fus/data/predicted_poses_uvfdata2.xlsx \
    --num_samples 30 \
    --window_stride 5 \
    --gpu $GPU

# Step 4: Evaluate
echo ""
echo "=== Step 4: Evaluate ==="
SKIP_FLAG=""
if [ "$SKIP_INR" = true ]; then
    SKIP_FLAG="--skip_inr"
fi
python bridge_nr_rec_fus/step4_evaluate.py \
    --gt_pose ./MyMoE/coords3d/uvfdata2.xlsx \
    --pred_pose ./bridge_nr_rec_fus/data/predicted_poses_uvfdata2.xlsx \
    --config ./configs/config_RGB_pose3d_uvfdata2.yaml \
    --images_dir ./MyMoE/uvfdata2 \
    --logdir_gt ./bridge_nr_rec_fus/results/inr_gt_poses \
    --logdir_pred ./bridge_nr_rec_fus/results/inr_pred_poses \
    --gpu $GPU \
    --num_epochs 100 \
    $SKIP_FLAG

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
