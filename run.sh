#! /bin/bash

BASEDIR=../../lab_resources/DDI/
export PYTHONPATH=$BASEDIR/util
export LD_LIBRARY_PATH=/home/martin/mambaforge/envs/mds/nvvm/libdevice:$LD_LIBRARY_PATH

# train NN
echo "Training NN"
XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/martin/mambaforge/envs/mds python3 train.py $BASEDIR/data/train $BASEDIR/data/devel mymodel

# run model on devel data and compute performance
echo "Predicting and evaluatig"
XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/martin/mambaforge/envs/mds python3 predict.py mymodel $BASEDIR/data/devel devel.out | tee devel.stats
