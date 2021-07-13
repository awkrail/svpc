model_type=$1  # [vivt, viv, vi, v]
model_path=$2 # /path/to/checkpoint

echo "---------------------------------------------------------"
echo ">>>>>>>> Running evaluating ingredient f1 on ${dset_name} dataset (test set)"
echo "---------------------------------------------------------"

python src/calculate_ingredient_f1.py \
    --model_name ${model_type} \
    --caption_path ${model_path}
