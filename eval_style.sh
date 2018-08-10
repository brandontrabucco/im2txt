#!/bin/bash

im2txt_env
python $SCRATCH/research/repos/im2txt/im2txt/evaluate_style.py \
--input_file_pattern="$SCRATCH/research/data/mscoco_dataset/val-?????-of-00004" \
--wikipedia_file_pattern="$SCRATCH/research/data/wikipedia_dataset/train-?????-of-00256" \
--checkpoint_dir="$SCRATCH/research/ckpts/im2txt/train/" \
--eval_dir="$SCRATCH/research/ckpts/im2txt/eval/" \
--annotations_file="$SCRATCH/research/data/mscoco_dataset/raw-data/annotations/captions_val2014.json"
