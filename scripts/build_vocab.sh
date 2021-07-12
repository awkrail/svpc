#!/usr/bin/env bash
# Usage:
# $ bash scripts/build_vocab.sh anet

glove_path=$1 # specify your glove.6B.300d.txt path

echo "---------------------------------------------------------"
echo ">>>>>>>> Running on YouCook2 Dataset"
min_word_count=3
train_path="./densevid_eval/our_yc2_data/yc2_train_anet_format.json"

python src/build_vocab.py \
--train_path ${train_path} \
--cache ./cache \
--min_word_count ${min_word_count} \
--raw_glove_path ${glove_path}
