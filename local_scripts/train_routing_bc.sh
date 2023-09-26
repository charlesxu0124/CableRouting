#!/bin/bash
EXP_NAME='{INSERT NAME HERE}'
OUTPUT_DIR='{INSERT PATH HERE}'
export PROJECT_HOME="$(pwd)"
export CONDA_OVERRIDE_CUDA="11.3"
export XLA_PYTHON_CLIENT_PREALLOCATE='false'
export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME/src"
export WANDB_API_KEY='{INSERT WANDB_API_KEY HERE}'


for IMAGE_AUG in 'rand'
do
    for WEIGHT_DECAY in 3e-3
    do
        python -m src.bc_main \
                    --dataset_path="./test_route.npy" \
                    --seed=24 \
                    --dataset_image_keys='wrist45_image:wrist225_image' \
                    --image_augmentation=${IMAGE_AUG} \
                    --total_steps=20000 \
                    --eval_freq=100 \
                    --batch_size=256 \
                    --save_model=True \
                    --lr=1e-3 \
                    --weight_decay=$WEIGHT_DECAY \
                    --policy_class_name="TanhGaussianResNetPolicy" \
                    --policy.spatial_aggregate='average' \
                    --policy.resnet_type='ResNet18' \
                    --policy.state_injection='z_only' \
                    --policy.share_resnet_between_views=False \
                    --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
                    --logger.online=True \
                    --logger.prefix='CableRouting' \
                    --logger.project="$EXP_NAME" \
                    --logger.random_delay=60.0

    done
done


