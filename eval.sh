module load python3/3.4.2
module load cuda/9.0
source $SCRATCH/research/envs/im2txt_env/bin/activate
export CUDA_VISIBLE_DEVICES=""

python3 $SCRATCH/research/repos/im2txt/im2txt/evaluate.py \
--input_file_pattern="$SCRATCH/research/data/coco/val-?????-of-00004" \
--wikipedia_file_pattern="$SCRATCH/research/data/wikipedia/train-?????-of-00256" \
--checkpoint_dir="$SCRATCH/research/ckpts/im2txt/train/" \
--eval_dir="$SCRATCH/research/ckpts/im2txt/eval/" \
--vocab_file="$SCRATCH/research/data/wikipedia/word_counts.txt" \
--annotations_file="$SCRATCH/research/data/coco/raw-data/annotations/captions_val2014.json"
