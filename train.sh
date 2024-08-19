source config.sh

python train.py \
    --task MultitaskOrthogonalTracenorm\
    --batch_size 512 \
    --epochs 600 \
    --p_dropout 0.25 \
    --input_dim 3 \
    --n_hidden_1 128 \
    --n_hidden_2 64 \
    --n_classes 12 \
    --learning_rate 0.001 \
    --lr_scheduler CosineAnnealingWarmRestarts \
    --w_regression 0.1 \
    --w_classify 0.5 \
    --w_grad 0.5 \
    --w_trace_norm 0.001 \
    --data_path $DATA_PATH \
    --output_dir models/mtt_orthogonal_tracenorm_1908 \
    --log_step 5 
    # --log_wandb \
    # --project_name "Experiments MTT" \
    # --experiment_name "MTT-Orthogonal-Tracenorm"\
    # --log_console 


