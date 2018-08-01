#!/bin/bash

im2txt_env
python $SCRATCH/research/repos/im2txt/im2txt/run_inference.py \
--checkpoint_path="$SCRATCH/research/ckpts/im2txt/train/" \
--input_files="$SCRATCH/research/data/mscoco_dataset/raw-data/val2014/COCO_val2014_000000224477.jpg" \
