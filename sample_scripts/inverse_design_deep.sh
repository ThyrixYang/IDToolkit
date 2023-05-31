# bash inverse_design_deep.sh <env_name> <eval_method> <deep inverse model>
python experiments/main.py \
    --env $1 \
    --eval_method $2 \
    --method $3 \
    --dataset_path datasets/$1 \
    --method_config experiments/configs/nn_default+$3 \
    --pred_num 100 \
    --train_num 1000000 