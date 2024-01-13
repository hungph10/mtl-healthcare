
python multitask_train.py \
    --batch_size 512 \
    --epochs 600 \
    --input_dim 3 \
    --n_hidden_1 128 \
    --n_hidden_2 64 \
    --n_classes 12 \
    --p_dropout 0.25 \
    --learning_rate 0.001 \
    --log_steps 5 \
    --data_path "data/res_and_16pos.npz" \
    --output_dir "models/multitask_LSTM" \
    --project_name="Multitask healthcare" \
    --experiment_name="mtl-LSTM-128-64" \
    --log_wandb 
