#!/bin/bash

# 使用MultiplicativeActivation的训练脚本
# 专家网络使用新的乘法激活函数，Manager网络仍使用sine激活函数

echo "开始训练RGBPose3D模型，使用MultiplicativeActivation激活函数"
echo "专家网络: multiplicative激活函数"
echo "Manager网络: sine激活函数"
echo "=========================================="

# 设置参数
CONFIG_FILE="configs/config_RGB_pose3d_manager_fixed_multiplicative.yaml"
LOG_DIR="./log/pose3d_multiplicative_$(date +%Y%m%d_%H%M%S)"
GPU_ID=0

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "配置文件: $CONFIG_FILE"
echo "日志目录: $LOG_DIR"
echo "GPU设备: $GPU_ID"
echo "=========================================="

# 运行训练
python image_reconstruction/train_rgb_pose3d_multiplicative.py \
    --config "$CONFIG_FILE" \
    --logdir "$LOG_DIR" \
    --gpu $GPU_ID

echo "训练完成！"
echo "结果保存在: $LOG_DIR"
