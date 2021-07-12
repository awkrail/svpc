#!/usr/bin/env bash
# Usage:
#   $ bash {SCRIPT.sh} {DATASET} [Any flags available in train.py, could also be empty]
#   DATASET: `anet` or `yc2`
#   Note the additional flags added will overwrite the specified flags below,
#   i.e., if `--exp_id run1` is specified, it will overwrite `--exp_id init` below.
# Examples:
#   anet debug mode: $ bash scripts/train.sh anet -debug
#   yc2 training mode: $ bash scripts/train.sh yc2

dset_name=$1  # [anet, yc2]
model_type=$2  # [mart, xl, xlrg, mtrans, mart_no_recurrence]
temperature=$3 # [0.1, 0.5, 1.0]
lam=$4 # [0.1, 0.5, 1.0]

data_dir="./densevid_eval/our_${dset_name}_data"
#v_feat_dir="/mnt/LSTA5/data/common/recipe/youcook2/features/yc2/${dset_name}_trainval"
v_feat_dir="/mnt/LSTA5/data/common/recipe/youcook2/features/training"
#v_feat_dir="/mnt/LSTA5/data/nishimura/graph_youcook2_generator/baselines/recurrent-transformer/features/training"
dur_file="/mnt/LSTA5/data/common/recipe/youcook2/features/yc2/${dset_name}_duration_frame.csv"
word2idx_path="./cache/${dset_name}_word2idx.json"
glove_path="./cache/${dset_name}_vocab_glove.pt"
#verb_glove_path="./cache/${dset_name}_verb_vocab_glove.pt"
verb_glove_path="./cache/bosselut_${dset_name}_verb_vocab_glove.pt"
model_path="/mnt/LSTA5/data/nishimura/graph_youcook2_generator/proposed_method/new_split_captioning/proposed_method/${model_type}_lambda_${lam}_tau_${temperature}"


echo "---------------------------------------------------------"
echo ">>>>>>>> Running training on ${dset_name} dataset"
if [[ ${dset_name} == "anet" ]]; then
    max_n_sen=6
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=100
elif [[ ${dset_name} == "yc2" ]]; then
    max_n_sen=12
    max_t_len=22  # including "BOS" and "EOS"
    max_v_len=100
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

echo ">>>>>>>> Model type ${model_type}"
echo "---------------------------------------------------------"
extra_args=()
if [[ ${model_type} == "full" ]]; then # w/ + reasoning + copy + repredict
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--full)
elif [[ ${model_type} == "reasoning" ]]; then # w/ + reasoning
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--reasoning)
elif [[ ${model_type} == "reason_copy" ]]; then # w/ reasoning + copy
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--reason_copy)
elif [[ ${model_type} == "ingr" ]]; then # w/ only ingredients
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--ingr)
elif [[ ${model_type} == "video" ]]; then # w/o any modules
    extra_args+=(--recurrent)
    extra_args+=(--ours)
    extra_args+=(--video)
else
    echo "Wrong option for your first argument, select between anet and yc2"
fi

python src/dump_memories.py \
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
