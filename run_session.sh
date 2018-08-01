#!/bin/bash

set -x 
cd "$SCRATCH/research/repos/im2txt"

./eval.sh &
./init.sh
./fine_tune.sh


