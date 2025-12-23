# TinyCMIF-VIO

This repository contains the reproduced and compressed models for [CMIF-VIO: A Novel Cross Modal Interaction Framework for Visual Inertial Odometry](https://ieeexplore.ieee.org/document/10777572). 

+ CMIF-VIO (`CMIF_model.py`): the reproduced model of CMIF-VIO.
+ SmallCMIF-VIO (`SmallCMIF_model.py`): a compressed model of CMIF-VIO.
+ TinyCMIF-VIO(`TinyCMIF_model.py`): a grayscale image version of SmallCMIF-VIO.

## Data Preparation

The code in this repository is tested on KITTI Odometry dataset. The IMU data after pre-processing is provided under `data/imus`. To download the images and poses, please run

    $cd data
    $source data_prep.sh 

## IMU data format

The IMU data has 6 dimentions: 

1. acceleration in x, i.e. in direction of vehicle front (m/s^2)
2. acceleration in y, i.e. in direction of vehicle left (m/s^2)
3. acceleration in z, i.e. in direction of vehicle top (m/s^2)
4. angular rate around x (rad/s)
5. angular rate around y (rad/s)
6. angular rate around z (rad/s)

## Download pretrainined models

Pretrained models are provided in [Link](https://drive.google.com/file/d/1mdhtF40PojJrkZm4VSiVSzuS-nWIqFdx/view?usp=sharing).

+ `flownets_bn_EPE2.459.pth.tar`: FlowNet encoder Please download them and place it under `pretrain_models` directory if you want to use it.

+ `pretrained_CMIF.pth`: CMIF-VIO
+ `pretrained_SmallCMIF.pth`: SmallCMIF-VIO
+ `pretrained_TinyCMIF.pth`: TinyCMIF-VIO

## Test the pretrained model

For CMIF-VIO:

    python3 test_CMIF.py \
    --data_dir 'PATH/TO/YOUR/KITTI/DATA' \
    --model 'PATH/TO/YOUR/PRETRAIN/MODELS/pretrained_CMIF.pth' \
    --gpu_ids '0' --experiment_name 'YOUR_EXPERIMENT_NAME'

For SmallCMIF-VIO: Change the model initialize in `test_FCN.py` to SmallCMIF_VIO

    python3 test_FCN.py \
    --data_dir 'PATH/TO/YOUR/KITTI/DATA' \
    --model 'PATH/TO/YOUR/PRETRAIN/MODELS/pretrained_SmallCMIF.pth' \
    --gpu_ids '0' --experiment_name 'YOUR_EXPERIMENT_NAME'

For TinyCMIF-VIO: Change the model initialize in `test_FCN.py` to TinyCMIF_VIO

    python3 test_FCN.py \
    --data_dir 'PATH/TO/YOUR/KITTI/DATA' \
    --model 'PATH/TO/YOUR/PRETRAIN/MODELS/pretrained_TinyCMIF.pth' \
    --use_grey_img \
    --gpu_ids '0' --experiment_name 'YOUR_EXPERIMENT_NAME'


The figures and error records will be generated under `./results/pretrained/files`.

