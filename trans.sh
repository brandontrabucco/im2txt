#!/bin/bash

im2txt_env
python $SCRATCH/research/repos/im2txt/im2txt/run_dual.py \
--checkpoint_path="$SCRATCH/research/ckpts/im2txt/train/" \
--attrigram_checkpoint_path="$SCRATCH/research/ckpts/attrigram/train/" \
--input_files="$SCRATCH/research/repos/im2txt/data/*.jpg" \
