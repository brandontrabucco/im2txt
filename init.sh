python2 im2txt/train.py \
--input_file_pattern="/home/btrabucco/research/im2txt/im2txt/data/coco/train-?????-of-00256" \
--inception_checkpoint_file="/home/btrabucco/research/im2txt/im2txt/data/inception_v3.ckpt" \
--train_dir="/home/btrabucco/research/im2txt/im2txt/train/" \
--train_inception=false \
--number_of_steps=1000 \
