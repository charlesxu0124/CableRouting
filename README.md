# CableRouting
### [Project Homepage](https://sites.google.com/view/cablerouting/home)

### [Dataset Page](https://sites.google.com/view/cablerouting/data)

## Installation

#### Clone the repository
```shell
git clone git@github.com:tan-liam/CableRouting.git
cd CableRouting
```

#### Install and use the included Ananconda environment
```shell
conda create -n cable python=3.10
conda activate cable
pip install -r requirements.txt

# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install torch
```

#### Edit the following scripts to put your wandb API key into the environment variable `WANDB_API_KEY`
* `pretrain_resnet_embedding.sh`
* `train_routing_bc.sh`
* `train_highlevel.sh`
* `finetune_highlevel.sh`

## Downloading the data
View [dataset page](https://sites.google.com/view/cablerouting/data) for more detailed format of the data.

### Routing Dataset
To pretrain the ResNet embedding and train routing policy, download the routing dataset [here](https://rail.eecs.berkeley.edu/datasets/cable_routing/routing_primitive_offline_dataset.zip) into the `data` folder and unzip it. 

Then run `process_route_trajectories.py` to process the trajectory data into a single file containing all the transitions. Set the --trajectory_path to the route_trajectories folder and the output_path to be where you want to output the new .npy file (data folder recommended).

This is the data you'll use to train routing.

The file will be called route_transitions.npy
 
### Primitive Selection Dataset
To train the highlevel policy, download the preprocessed dataset [here](https://rail.eecs.berkeley.edu/datasets/cable_routing/end_to_end_trajectory_dataset.zip) into the `data` folder. Unzip the file and proceed to the next step. No additional processing is required. 

This is the data you'll use to train the high-level policy.

## Train the model
You can train the models with the following scripts.
```shell
local_scripts/pretrain_resnet_embedding.sh
local_scripts/train_routing_bc.sh
local_scripts/train_highlevel.sh
local_scripts/finetune_highlevel.sh
```

`pretrain_resnet_embedding.sh` will use the routing data to pretrain the ResNet. Please pass the path to `route_transitions` to the `dataset_path` flag. It will output a `model.pkl` file

`train_routing_bc.sh` will train the routing policy. Please pass the path to `route_transitions` to the `dataset_path` flag. It will output a `model.pkl` file.

`train_highlevel.sh` will train the high-level policy. You will need to pass in the trained model from `pretrain_resnet_embedding.sh` to the `encoder_checkpoint_path` flag. Please pass the path to `primitive_selection_offline_dataset.npy` to the `dataset_path` flag. This will output multiple `model.pkl` files at different checkpoints.

`finetune_highlevel.sh` will fine tune the high-level policy. You will need to pass in the output from `pretrain_resnet_embedding.sh` to the `encoder_checkpoint_path` flag. You will need to pass in the output from `train_highlevel.sh` into the `primitive_policy_checkpoint_path` flag. Choose an appropriate checkpoint. Please pass in the finetuning high-level data to the `dataset_path` flag. This will output multiple `model.pkl` files at different checkpoints.

## Visualize Experiment Results with W&B
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site).
To log to W&B, you first need to set your W&B API key environment variable:
```shell
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```
Then you can run experiments with W&B logging turned on from any of the .sh scripts:
```shell
--logger.online=True
```

## Citation BibTex

If you found this code useful, consider citing the following paper:
```
@article{luo2023multistage,
  author    = {Jianlan Luo and Charles Xu and Xinyang Geng and Gilbert Feng and Kuan Fang and Liam Tan and Stefan Schaal and Sergey Levine},
  title     = {Multi-Stage Cable Routing through Hierarchical Imitation Learning},
  journal   = {arXiv pre-print},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.08927},
}
```
# CableRouting
