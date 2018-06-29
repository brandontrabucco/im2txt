module load python3/3.4.2
module load cuda/9.0
source $SCRATCH/research/envs/im2txt_env/bin/activate

python3 $SCRATCH/research/repos/im2txt/im2txt/evaluate.py \
--input_file_pattern="$SCRATCH/research/data/coco/val-?????-of-00004" \
--checkpoint_dir="$SCRATCH/research/ckpts/im2txt/train/" \
--eval_dir="$SCRATCH/research/ckpts/im2txt/eval/" \
