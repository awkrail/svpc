#!/usr/bin/env bash

dset_name="yc2"
model_type=$1  # [vivt, viv, vi, v]
temperature=$2 # [0.1, 0.25, 0.5, 0.75, 1.0]
lam=$3 # [0.1, 0.25, 0.5, 0.75, 1.0]
model_path=$4 # /path/to/checkpoint
v_feat_dir=$5 # /path/to/feature/
dur_file=$6 # /path/to/duration_frame.csv

data_dir="./densevid_eval/${dset_name}_data"
#v_feat_dir="/mnt/LSTA5/data/common/recipe/youcook2/features/training"
#dur_file="/mnt/LSTA5/data/common/recipe/youcook2/features/yc2/${dset_name}_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
verb_glove_path="./cache/bosselut_${dset_name}_verb_vocab_glove.pt"
#model_path="/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/debbued_version/${model_type}_lambda_${lam}_tau_${temperature}"


echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset"
max_n_sen=12
max_t_len=22  # including "BOS" and "EOS"
max_v_len=100

echo ">>>>>>>> Model type ${model_type}"
echo "---------------------------------------------------------"
extra_args=()
if [[ ${model_type} == "vivt" ]]; then # w/ + reasoning + copy + repredict
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--full)
elif [[ ${model_type} == "viv" ]]; then # w/ reasoning + copy
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--reason_copy)
elif [[ ${model_type} == "vi" ]]; then # w/ only copy
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--copy)
elif [[ ${model_type} == "v" ]]; then # w/o any modules
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--video)
else
    echo "Wrong option for your first argument, select from [vivt, viv, vi, v]"
fi

python src/train.py \
--dset_name ${dset_name} \
--data_dir ${data_dir} \
--video_feature_dir ${v_feat_dir} \
--v_duration_file ${dur_file} \
--save_model ${model_path} \
--word2idx_path ${word2idx_path} \
--glove_path ${glove_path} \
--verb_glove_path ${verb_glove_path} \
--temperature ${temperature} \
--lam ${lam} \
--max_n_sen ${max_n_sen} \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
--exp_id init \
${extra_args[@]} \
