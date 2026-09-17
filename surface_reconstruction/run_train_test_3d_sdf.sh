#!/bin/bash

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/data/qty/anaconda3/envs/inr_moe/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="/data/qty/anaconda3/envs/inr_moe"

# 确保符号链接存在
libcuda_source="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
libcuda_dest="/data/qty/anaconda3/envs/inr_moe/lib/libcuda.so"

if [ ! -f "$libcuda_dest" ]; then
    ln -sf $libcuda_source $libcuda_dest
    echo "Created symlink: $libcuda_dest -> $libcuda_source"
fi

echo "PyKeOps environment setup complete"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"



GPU=0

export PYTHONUNBUFFERED=1
IDENTIFIER='my_3dsdf_recon_experiment'
LOGDIR='./log/'
CONFIG='../configs/config_3D.yaml'
for SHAPE in 'gt_armadillo.xyz' 'gt_dragon.xyz' 'gt_lucy.xyz'  'gt_thai.xyz'
do
  IDENTIFIER_S=${IDENTIFIER}'_'${SHAPE}
  CUDA_VISIBLE_DEVICES=0 python3 train_3d_shape.py --logdir $LOGDIR --config $CONFIG --shape_id $SHAPE --gpu $GPU --identifier $IDENTIFIER_S
done