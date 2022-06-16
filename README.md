
# Grasp Detection

## Introduction
This repository is code release for our grasp detection paper.

In this repository, we provide implementation of the proposed method (with Pytorch):
1. VoteGrasp model can be found in [models/votegrasp.py](https://github.com/hoangcuongbk80/NovelVoteGrasp/blob/master/models/votegrasp.py)
2. Context learning module and grasp generation module can be found in [models/proposal_module.py](https://github.com/hoangcuongbk80/NovelVoteGrasp/blob/master/models/proposal_module.py)
3. Loss function can found in [models/loss_helper.py](https://github.com/hoangcuongbk80/NovelVoteGrasp/blob/master/models/loss_helper.py)

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. The code is tested with Ubuntu 18.04, Pytorch v1.16, TensorFlow v1.14, CUDA 10.1 and cuDNN v7.4.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Dataset

Visit the [GraspNet](https://graspnet.net/datasets.html) Website to get the dataset.

#### Data preparation

Prepare data by running `python prepare_data.py --gen_data`

## Training and Predicting

#### Train

To train a new VoteGrasp model:

    CUDA_VISIBLE_DEVICES=0 python train.py --log_dir log_votegrasp

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options.
While training you can check the `log_votegrasp/log_train.txt` file on its progress.

#### Run predict

    python predict.py

## Acknowledgements
Will be available after our paper has been published.

## License
Will be available after our paper has been published.
