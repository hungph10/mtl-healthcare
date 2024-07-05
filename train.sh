

# # # Training multitask 
# python train_multitask.py \
#     --batch_size 1024 \
#     --epochs 600 \
#     --input_dim 3 \
#     --n_hidden_1 128 \
#     --n_hidden_2 64 \
#     --n_classes 12 \
#     --p_dropout 0.25 \
#     --learning_rate 0.001 \
#     --seed 42 \
#     --log_steps 5 \
#     --w_regression 0.5 \
#     --w_classify 0.5 \
#     --data_path "data/multitask_cls12_regr.npz" \
#     --output_dir "models/test" \
#     --project_name "Multitask healthcare" \
#     --experiment_name "mtl-LSTM-128-64-orthogonal" \
#     # --log_wandb



#  python train_kan_multitask_orthogonal.py \
#     --batch_size 1024 \
#     --epochs 600 \
#     --input_dim 3 \
#     --n_hidden_1 128 \
#     --n_hidden_2 64 \
#     --n_classes 12 \
#     --p_dropout 0.25 \
#     --learning_rate 0.001 \
#     --seed 42 \
#     --log_steps 5 \
#     --w_regression 0.1 \
#     --w_classify 0.5 \
#     --w_grad 0.5 \
#     --data_path "data/multitask_cls12_regr.npz" \
#     --output_dir "models/multitask_LSTM_orthogonal" \
#     --project_name "MTL - Experiments" \
#     --experiment_name "mtl-LSTM-128-64-orthogonal-reg0.1-cls0.5-grad0.5" \

python train_classify.py \
    --batch_size 512 \
    --epochs 600 \
    --input_dim 3 \
    --n_hidden_1 128 \
    --n_hidden_2 64 \
    --n_classes 12 \
    --p_dropout 0.25 \
    --learning_rate 0.001 \
    --seed 42 \
    --log_steps 5 \
    --data_path "data/multitask_cls12_regr.npz" \
    --output_dir "models/single_task_classify" \
    --project_name "experiments-research" \
    --experiment_name "single-task-LSTM-128-64"