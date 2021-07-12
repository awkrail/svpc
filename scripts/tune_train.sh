#!/usr/bin/env bash
# Usage:
#   $ bash {SCRIPT.sh} {DATASET} [Any flags available in train.py, could also be empty]
#   DATASET: `anet` or `yc2`
#   Note the additional flags added will overwrite the specified flags below,
#   i.e., if `--exp_id run1` is specified, it will overwrite `--exp_id init` below.
# Examples:
#   anet debug mode: $ bash scripts/train.sh anet -debug
#   yc2 training mode: $ bash scripts/train.sh yc2

temperature=$1 # [0.1, 0.5, 1.0]
lam=$2 # [0.1, 0.5, 1.0]

python src/train.py --dset_name yc2 --data_dir ./densevid_eval/our_yc2_data --video_feature_dir /mnt/LSTA5/data/common/recipe/youcook2/features/training --v_duration_file /mnt/LSTA5/data/common/recipe/youcook2/features/yc2/yc2_duration_frame.csv --save_model /mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/captioning/tuning/lambda_${temperature}_tau_${lam}_bilstm_full --word2idx_path ./cache/yc2_word2idx.json --glove_path ./cache/yc2_vocab_glove.pt --verb_glove_path ./cache/bosselut_yc2_verb_vocab_glove.pt --max_n_sen 12 --max_t_len 22 --max_v_len 100 --temperature $1 --lam $2 --exp_id init --recurrent --ours --full
