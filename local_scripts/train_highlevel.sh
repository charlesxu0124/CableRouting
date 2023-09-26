#! /bin/bash
EXP_NAME='{INSERT NAME HERE}'
OUTPUT_DIR='{INSERT DIR HERE}'
export PROJECT_HOME="$(pwd)"
export CONDA_OVERRIDE_CUDA="11.3"
export XLA_PYTHON_CLIENT_PREALLOCATE='false'
export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME/src"
export WANDB_API_KEY='{INSERT WANDB_API_KEY}'

python -m src.primitive_selection_main \
            --encoder_checkpoint_path="{INSERT PATH HERE}" \
            --dataset_path="{INSERT PATH HERE}"\
            --seed=24 \
            --dataset_image_keys='wrist45_image:wrist225_image:side_image' \
            --image_augmentation='rand' \
            --total_steps=30000 \
            --eval_freq=100 \
            --batch_size=128 \
            --save_model=True \
            --lr=3e-4 \
            --weight_decay=1e-2 \
            --policy.spatial_aggregate='average' \
            --policy.resnet_type='ResNet18' \
            --policy.state_injection='z_only' \
            --policy.share_resnet_between_views=False \
            --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logger.online=True \
            --logger.prefix='CableRouting' \
            --logger.project="$EXP_NAME" 
 
 
