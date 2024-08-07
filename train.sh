# Train classify
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
    --data_path "data/update-0208/processed_and_augmented.npz" \
    --output_dir "models/classify_0708" \
    # --log_console
    # --log_wabdb \
    # --project_name "experiments-research" \
    # --experiment_name "single-task-LSTM-128-64"


# # Train regression
# python train_regression.py \
#     --batch_size 512 \
#     --epochs 600 \
#     --input_dim 3 \
#     --n_hidden_1 128 \
#     --n_hidden_2 64 \
#     --p_dropout 0.25 \
#     --learning_rate 0.001 \
#     --seed 42 \
#     --log_steps 5 \
#     --data_path "data/update-0208/processed_and_augmented.npz" \
#     --output_dir "models/regression_0708" \
#     # --log_console
#     # --log_wabdb \
#     # --project_name "experiments-research" \
#     # --experiment_name "single-task-LSTM-128-64"


# # Training multitask base
# python train_multitask.py \
#     --batch_size 512 \
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
#     --data_path "data/update-0208/processed_and_augmented.npz" \
#     --output_dir "models/multitask_base_0708" \
#     # --log_console
#     # --log_wabdb \
#     # --project_name "experiments-research" \
#     # --experiment_name "multitask-base"


# # Training multitask + orthogonal gradient
# python train_multitask_orthogonal.py \
#     --batch_size 512 \
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
#     --data_path "data/update-0208/processed_and_augmented.npz" \
#     --output_dir "models/multitask_orthogonal_0708" \
#     # --log_console
#     # --log_wabdb \
#     # --project_name "experiments-research" \
#     # --experiment_name "multitask-base"


# # Training multitask + orthogonal gradient + tracenorm
# python train_multitask_orthogonal_tracenorm.py \
#     --batch_size 512 \
#     --epochs 600 \
#     --input_dim 3 \
#     --n_hidden_1 128 \
#     --n_hidden_2 64 \
#     --n_classes 12 \
#     --p_dropout 0.25 \
#     --learning_rate 0.001 \
#     --seed 42 \
#     --log_steps 5 \
#     --w_regression 0.4 \
#     --w_classify 0.9 \
#     --w_grad 0.3 \
#     --w_trace_norm 0.001 \
#     --data_path "data/update-0208/processed_and_augmented.npz" \
#     --output_dir "models/multitask_orthogonal_tracenorm_0708" \
#     # --log_console
#     # --log_wabdb \
#     # --project_name "experiments-research" \
#     # --experiment_name "multitask-base"

