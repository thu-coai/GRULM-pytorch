[![Main Repo](https://img.shields.io/badge/Main_project-cotk-blue.svg?logo=github)](https://github.com/thu-coai/cotk)
[![This Repo](https://img.shields.io/badge/Model_repo-GRULM--pytorch-blue.svg?logo=github)](https://github.com/thu-coai/GRULM-pytorch)
[![Coverage Status](https://coveralls.io/repos/github/thu-coai/GRULM-pytorch/badge.svg?branch=master)](https://coveralls.io/github/thu-coai/GRULM-pytorch?branch=master)
[![Build Status](https://travis-ci.com/thu-coai/GRULM-pytorch.svg?branch=master)](https://travis-ci.com/thu-coai/GRULM-pytorch)

This repo is a benchmark model for [CoTK](https://github.com/thu-coai/cotk) package.

# GRULM (PyTorch)

A GRU language model in pytorch.

You can refer to the following paper for details:

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.

## Require Packages

* **python3**
* CoTK >= 0.1.0
* pytorch >= 1.0.0
* tensorboardX >= 1.4

## Quick Start

* Install ``CoTK`` following [instructions](https://github.com/thu-coai/cotk#installation).
* Using ``cotk download thu-coai/GRULM-pytorch/master`` to download codes.
* Execute ``python run.py`` to train the model.
  * The default dataset is ``resources://MSCOCO``. You can use ``--dataid`` to specify data path (can be a local path, a url or a resources id). For example: ``--dataid /path/to/datasets``
  * It doesn't use pretrained word vector by default setting. You can use ``--wvid`` to specify data path for pretrained word vector (can be a local path, a url or a resources id). For example: ``--wvid resources://Glove300``
  * If you don't have GPUs, you can add `--cpu` for switching to CPU, but it may cost very long time for either training or test.
* You can view training process by tensorboard, the log is at `./tensorboard`.
  * For example, ``tensorboard --logdir=./tensorboard``. (You have to install tensorboard first.)
* After training, execute  ``python run.py --mode test --restore best`` for test.
  * You can use ``--restore filename`` to specify checkpoints files, which are in ``./model``. For example: ``--restore pretrained-mscoco`` for loading ``./model/pretrained-mscoco.model``
  * ``--restore last`` means last checkpoint, ``--restore best`` means best checkpoints on dev.
  * ``--restore NAME_last`` means last checkpoint with model named NAME. The same as``--restore NAME_best``.
* Find results at ``./output``.

## Arguments

```none
usage: run.py [-h] [--name NAME] [--restore RESTORE] [--mode MODE]
              [--dh_size DH_SIZE] [--droprate DROPRATE]
              [--decode_mode {max,sample,gumbel,samplek,beam}] [--batchnorm]
              [--top_k TOP_K] [--length_penalty LENGTH_PENALTY]
              [--temperature TEMPERATURE] [--dataid DATAID] [--epoch EPOCH]
              [--batch_per_epoch BATCH_PER_EPOCH] [--wvid WVID]
              [--out_dir OUT_DIR] [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
              [--cache_dir CACHE_DIR] [--cpu] [--debug] [--cache]
              [--seed SEED] [--lr LR]

A language model with GRU. Attention, beamsearch, dropout and batchnorm is
supported.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name of your model, used for tensorboard, etc.
                        Default: runXXXXXX_XXXXXX (initialized by current
                        time)
  --restore RESTORE     Checkpoints name to load. "NAME_last" for the last
                        checkpoint of model named NAME. "NAME_best" means the
                        best checkpoint. You can also use "last" and "best",
                        by default use last model you run. Attention:
                        "NAME_last" and "NAME_best" are not guaranteed to work
                        when 2 models with same name run in the same time.
                        "last" and "best" are not guaranteed to work when 2
                        models run in the same time. Default: None (don't load
                        anything)
  --mode MODE           "train" or "test". Default: train
  --dh_size DH_SIZE     Size of decoder GRU
  --droprate DROPRATE   The probability to be zeroed in dropout. 0 indicates
                        for don't use dropout
  --decode_mode {max,sample,gumbel,samplek,beam}
                        The decode strategy when freerun. Choices: max,
                        sample, gumbel(=sample), samplek(sample from topk),
                        beam(beamsearch). Default: samplek
  --batchnorm           Use bathnorm
  --top_k TOP_K         The top_k when decode_mode == "beam" or "samplek"
  --length_penalty LENGTH_PENALTY
                        The beamsearch penalty for short sentences. The
                        penalty will get larger when this becomes smaller.
  --temperature TEMPERATURE
                        Temperature. Default: 1
  --dataid DATAID       Resources/path for data set. Default:
                        resources://MSCOCO
  --epoch EPOCH         Epoch for training. Default: 100
  --batch_per_epoch BATCH_PER_EPOCH
                        Batches per epoch. Default: 1500
  --wvid WVID           Resources/path for pretrained wordvector. Default:
                        resources://Glove300d
  --out_dir OUT_DIR     Output directory for test output. Default: ./output
  --log_dir LOG_DIR     Log directory for tensorboard. Default: ./tensorboard
  --model_dir MODEL_DIR
                        Checkpoints directory for model. Default: ./model
  --cache_dir CACHE_DIR
                        Checkpoints directory for cache. Default: ./cache
  --cpu                 Use cpu.
  --debug               Enter debug mode (using ptvsd).
  --cache               Use cache for speeding up load data and wordvec. (It
                        may cause problems when you switch dataset.)
  --seed SEED           Specify random seed. Default: 0
  --lr LR               Learning rate. Default: 0.001
```

## Example

WAIT FOR UPDATE

## Performance

WAIT FOR UPDATE

## Author

[HUANG Fei](https://github.com/hzhwcmhf)
