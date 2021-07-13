#!/usr/bin/env bash

dset_name="yc2"
model_type=$1  # [vivt, viv, vi, v]
model_path=$2 # /path/to/checkpoint
v_feat_dir=$3 # /path/to/feature/
dur_file=$4 # /path/to/duration_frame.csv

data_dir="./densevid_eval/${dset_name}_data"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
verb_glove_path="./cache/bosselut_${dset_name}_verb_vocab_glove.pt"

echo "---------------------------------------------------------"
echo ">>>>>>>> Running evaluating on ${dset_name} dataset (test set)"
max_n_sen=12
max_t_len=22  # including "BOS" and "EOS"
max_v_len=100

echo ">>>>>>>> Model type ${model_type}"
echo "---------------------------------------------------------"
extra_args=()
if [[ ${model_type} == "vivt" ]]; then # VIVT
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--full)
elif [[ ${model_type} == "viv" ]]; then # VIV
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--reason_copy)
elif [[ ${model_type} == "vi" ]]; then # VI
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--copy)
elif [[ ${model_type} == "v" ]]; then # V
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--video)
else
    echo "Wrong option for your first argument, select from [vivt, viv, vi, v]"
fi

python src/test.py \
--dset_name ${dset_name} \
--data_dir ${data_dir} \
--video_feature_dir ${v_feat_dir} \
--v_duration_file ${dur_file} \
--save_model ${model_path} \
--word2idx_path ${word2idx_path} \
--glove_path ${glove_path} \
--verb_glove_path ${verb_glove_path} \
--max_n_sen ${max_n_sen} \
--max_t_len ${max_t_len} \
--max_v_len ${max_v_len} \
--exp_id init \
${extra_args[@]} \
