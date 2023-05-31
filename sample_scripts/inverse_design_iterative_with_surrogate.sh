# bash inverse_design_iterative_with_surrogate.sh <env_name> <eval_method> <deep inverse model> <substitute model>
python experiments/main.py \
    --env $1 \
    --eval_method $2 \
    --method $3 \
    --dataset_path datasets/$1 \
    --method_config experiments/configs/$3 \
    --pred_num 10000 \
    --train_num 0 \
    --substitute_model $4 \
    --ensemble \
    --evaluate_after_search