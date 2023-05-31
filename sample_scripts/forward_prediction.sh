# bash forward_prediction.sh <env_name> <nn_forward_model> <sklearn_forward_model>

# train a pytorch lightning interface based forward prediction model
python experiments/main.py \
    --method neural_network \
    --eval_method forward_pred \
    --env $1 \
    --dataset_path datasets/$1 \
    --method_config experiments/configs/nn_default+$2 \
    --save_substitute_model \
    --seeds "0,1"

# train a sklearn interface based forward prediction model
python experiments/main.py \
    --method sklearn \
    --eval_method forward_pred \
    --env $1 \
    --dataset_path datasets/$1 \
    --method_config experiments/configs/$3 \
    --save_substitute_model \
    --seeds "0,1"