#!/bin/bash

im2txt_env
python $SCRATCH/research/repos/im2txt/im2txt/train.py \
--input_file_pattern="$SCRATCH/research/data/mscoco_dataset/train-?????-of-00256" \
--wikipedia_file_pattern="$SCRATCH/research/data/wikipedia_dataset/train-?????-of-00256" \
--inception_checkpoint_file="$SCRATCH/research/ckpts/inception/inception_v3.ckpt" \
--train_dir="$SCRATCH/research/ckpts/im2txt/train/" \
--train_inception=false \
--number_of_steps=50000 \
