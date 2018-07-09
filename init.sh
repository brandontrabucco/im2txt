module load python3/3.4.2
module load cuda/9.0
source $SCRATCH/research/envs/im2txt_env/bin/activate

python3 $SCRATCH/research/repos/im2txt/im2txt/train.py \
--input_file_pattern="$SCRATCH/research/data/coco/train-?????-of-00256" \
--wikipedia_file_pattern="$SCRATCH/research/data/wikipedia/train-?????-of-00256" \
--inception_checkpoint_file="$SCRATCH/research/ckpts/inception/inception_v3.ckpt" \
--train_dir="$SCRATCH/research/ckpts/im2txt/train/" \
--train_inception=false \
--number_of_steps=100000 \
