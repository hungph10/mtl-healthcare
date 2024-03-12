# Training multitask
python train_multitask.py \
    --batch_size 256 \
    --epochs 600 \
    --input_dim 3 \
    --n_hidden_1 128 \
    --n_hidden_2 64 \
    --n_classes 12 \
    --p_dropout 0.25 \
    --learning_rate 0.001 \
    --fix_random False \
    --log_steps 5 \
    --data_path "/mnt/new2/acadimic/mtl-healthcare/data/respiration_regression_and_12posture-classification.npz" \
    --output_dir "models/multitask_LSTM" \
    --project_name="Multitask healthcare" \
    --experiment_name="mtl-LSTM-128-64" \
    --log_wandb


# Training multitask using orthogonal
python train_multitask_orthogonal.py \
    --batch_size 256 \
    --epochs 600 \
    --input_dim 3 \
    --n_hidden_1 128 \
    --n_hidden_2 64 \
    --n_classes 12 \
    --p_dropout 0.25 \
    --learning_rate 0.001 \
    --fix_random False \
    --log_steps 5 \
    --weight_regression 1 \
    --weight_classify 1 \
    --weight_grad 1 \
    --data_path "data/multitask_cls12_regr.npz" \
    --output_dir "models/multitask_LSTM" \
    --project_name="Multitask healthcare" \
    --experiment_name="mtl-LSTM-128-64-orthogonal" \
    --log_wandb
