
PROJECT_NAME="Experiment accelerometer signal multitask learning"
DATA_PATH="/kaggle/input/official-data/train_val_test-1508.npz"

# Grid search
SEEDS=(42)
TASKS=("Classify" "Regression" "Multitask" "MultitaskOrthogonal" "MultitaskOrthogonalTracenorm")
BATCH_SIZE=(512)
EPOCHS=(600)
DROPOUT=(0.25)

LEARNING_RATE=(0.001)
LR_SCHEDULER=("CosineAnnealingWarmRestarts" "StepLR")
W_REGRESSION=(0.1)
W_CLASSIFY=(0.5)
W_GRAD=(0.5)
W_TRACENORM=(0.001)


# Hyper parameters for LSTM architecture
HIDDEN_SIZE_LSTM1=(128)
HIDDEN_SIZE_LSTM2=(64)

# Hyper parameters for CNN-Attention architecture
HIDDEN_SIZE_CONV1=(64)
HIDDEN_SIZE_CONV2=(128)
HIDDEN_SIZE_CONV3=(256)
KERNEL_SIZE=(3)
NUM_HEADS=(16)


TASKS=("MultitaskOrthogonalTracenorm")

# Loop through each task and run the training script
for seed in "${SEEDS[@]}"
    for batch_size in "${BATCH_SIZE[@]}"
        for epochs in "${EPOCHS[@]}"
            for p_dropout in "${DROPOUT[@]}"
                for lr in "${LEARNING_RATE[@]}"
                    for lr_scheduler in "${LR_SCHEDULER[@]}"
                        for network in "${NETWORK[@]}"
                            if ["$network" == "LSTM"]
                            then
                                for hidden_size_lstm1 in "${HIDDEN_SIZE_LSTM1[@]}"
                                    for hidden_size_lstm2 in "${HIDDEN_SIZE_LSTM2[@]}"
                                        for task in "${TASKS[@]}"
                                            do
                                                if [["$task" == "Classify"] || ["$task" == "Regression"]]
                                                then
                                                    EXP_NAME="$network-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-learning_rate_$lr-lr_scheduler_$lr_scheduler" \
                                                    echo "Running: $EXP_NAME"
                                                    python train.py \
                                                        --seed $seed \
                                                        --task $task\
                                                        --batch_size $batch_size \
                                                        --epochs $epochs \
                                                        --p_dropout $p_dropout \
                                                        --network LSTM \
                                                        --input_dim 3 \
                                                        --hidden_size_lstm1 $hidden_size_lstm1 \
                                                        --hidden_size_lstm2 $hidden_size_lstm2 \
                                                        --n_classes 12 \
                                                        --learning_rate $lr \
                                                        --lr_scheduler $lr_scheduler \
                                                        --data_path "$DATA_PATH" \
                                                        --output_dir "models/$EXP_NAME" \
                                                        --log_steps 5 \
                                                        --log_wandb \
                                                        --project_name "$PROJECT_NAME" \
                                                        --experiment_name "$EXP_NAME"
                                                elif ["$task" == "Multitask"]
                                                then
                                                    for w_classify in "${W_CLASSIFY[@]}"
                                                        for w_regression in "${W_REGRESSION[@]}" 
                                                            do
                                                                EXP_NAME="$network-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-learning_rate_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify" \
                                                                python train.py \
                                                                    --seed $seed \
                                                                    --task $task\
                                                                    --batch_size $batch_size \
                                                                    --epochs $epochs \
                                                                    --p_dropout $p_dropout \
                                                                    --network LSTM \
                                                                    --input_dim 3 \
                                                                    --hidden_size_lstm1 $hidden_size_lstm1 \
                                                                    --hidden_size_lstm2 $hidden_size_lstm2 \
                                                                    --n_classes 12 \
                                                                    --learning_rate $lr \
                                                                    --lr_scheduler $lr_scheduler \
                                                                    --w_regression $w_regression \
                                                                    --w_classify $w_classify \
                                                                    --data_path "$DATA_PATH" \
                                                                    --output_dir "models/$EXP_NAME" \
                                                                    --log_steps 5 \
                                                                    --log_wandb \
                                                                    --project_name "$PROJECT_NAME" \
                                                                    --experiment_name "$EXP_NAME"
                                                            done
                                                elif ["$task" == "MultitaskOrthogonal"]
                                                then
                                                    for w_classify in "${W_CLASSIFY[@]}"
                                                        for w_regression in "${W_REGRESSION[@]}" 
                                                            for w_grad in "${W_GRAD[@]}" 
                                                                do
                                                                    EXP_NAME="$network-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-learning_rate_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify-w_grad_$w_grad" \
                                                                    python train.py \
                                                                        --seed $seed \
                                                                        --task $task\
                                                                        --batch_size $batch_size \
                                                                        --epochs $epochs \
                                                                        --p_dropout $p_dropout \
                                                                        --network LSTM \
                                                                        --input_dim 3 \
                                                                        --hidden_size_lstm1 $hidden_size_lstm1 \
                                                                        --hidden_size_lstm2 $hidden_size_lstm2 \
                                                                        --n_classes 12 \
                                                                        --learning_rate $lr \
                                                                        --lr_scheduler $lr_scheduler \
                                                                        --w_regression $w_regression \
                                                                        --w_classify $w_classify \
                                                                        --w_grad $w_grad \
                                                                        --data_path "$DATA_PATH" \
                                                                        --output_dir "models/$EXP_NAME" \
                                                                        --log_steps 5 \
                                                                        --log_wandb \
                                                                        --project_name "$PROJECT_NAME" \
                                                                        --experiment_name "$EXP_NAME"
                                                                done
                                                elif ["$task" == "MultitaskOrthogonalTracenorm"]
                                                then
                                                    for w_classify in "${W_CLASSIFY[@]}"
                                                        for w_regression in "${W_REGRESSION[@]}" 
                                                            for w_grad in "${W_GRAD[@]}" 
                                                                for w_trace_norm in "${W_TRACENORM[@]}" 
                                                                    do
                                                                        python train.py \
                                                                        --seed $seed \
                                                                        --task $task\
                                                                        --batch_size $batch_size \
                                                                        --epochs $epochs \
                                                                        --p_dropout $p_dropout \
                                                                        --network LSTM \
                                                                        --input_dim 3 \
                                                                        --hidden_size_lstm1 $hidden_size_lstm1 \
                                                                        --hidden_size_lstm2 $hidden_size_lstm2 \
                                                                        --n_classes 12 \
                                                                        --learning_rate $lr \
                                                                        --lr_scheduler $lr_scheduler \
                                                                        --w_regression $w_regression \
                                                                        --w_classify $w_classify \
                                                                        --w_grad $w_grad \
                                                                        --w_trace_norm $w_trace_norm \
                                                                        --data_path "$DATA_PATH" \
                                                                        --output_dir "models/$EXP_NAME" \
                                                                        --log_steps 5 \
                                                                        --log_wandb \
                                                                        --project_name "$PROJECT_NAME" \
                                                                        --experiment_name "$EXP_NAME"
                                                                    done
                                                fi
                                            done
                            elif ["$network" == "CNN-Attention"]
                            then
                                for hidden_size_conv1 in "${HIDDEN_SIZE_CONV1[@]}"
                                    for hidden_size_conv2 in "${HIDDEN_SIZE_CONV2[@]}"
                                        for hidden_size_conv3 in "${HIDDEN_SIZE_CONV3[@]}"
                                            for kernel_size in "${KERNEL_SIZE[@]}"
                                                for num_heads in "${NUM_HEADS[@]}"
                                                    for task in "${TASKS[@]}"
                                                        do
                                                            if [["$task" == "Classify"] || ["$task" == "Regression"]]
                                                            then
                                                                EXP_NAME="$network-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv1_$hidden_size_conv1-hidden_size_conv2_$hidden_size_conv2-hidden_size_conv3_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-learning_rate_$lr-lr_scheduler_$lr_scheduler" \
                                                                echo "Running: $EXP_NAME"
                                                                python train.py \
                                                                    --seed $seed \
                                                                    --task $task\
                                                                    --batch_size $batch_size \
                                                                    --epochs $epochs \
                                                                    --p_dropout $p_dropout \
                                                                    --network LSTM \
                                                                    --input_dim 3 \
                                                                    --hidden_size_conv1 $hidden_size_conv1 \
                                                                    --hidden_size_conv2 $hidden_size_conv2 \
                                                                    --hidden_size_conv3 $hidden_size_conv3 \
                                                                    --kernel_size $kernel_size \
                                                                    --num_heads $num_heads \
                                                                    --n_classes 12 \
                                                                    --learning_rate $lr \
                                                                    --lr_scheduler $lr_scheduler \
                                                                    --data_path "$DATA_PATH" \
                                                                    --output_dir "models/$EXP_NAME" \
                                                                    --log_steps 5 \
                                                                    --log_wandb \
                                                                    --project_name "$PROJECT_NAME" \
                                                                    --experiment_name "$EXP_NAME"
                                                            elif ["$task" == "Multitask"]
                                                            then
                                                                for w_classify in "${W_CLASSIFY[@]}"
                                                                    for w_regression in "${W_REGRESSION[@]}" 
                                                                        do
                                                                            EXP_NAME="$network-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv1_$hidden_size_conv1-hidden_size_conv2_$hidden_size_conv2-hidden_size_conv3_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-learning_rate_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify" \
                                                                            python train.py \
                                                                                --seed $seed \
                                                                                --task $task\
                                                                                --batch_size $batch_size \
                                                                                --epochs $epochs \
                                                                                --p_dropout $p_dropout \
                                                                                --network LSTM \
                                                                                --input_dim 3 \
                                                                                --hidden_size_conv1 $hidden_size_conv1 \
                                                                                --hidden_size_conv2 $hidden_size_conv2 \
                                                                                --hidden_size_conv3 $hidden_size_conv3 \
                                                                                --kernel_size $kernel_size \
                                                                                --num_heads $num_heads \
                                                                                --n_classes 12 \
                                                                                --learning_rate $lr \
                                                                                --lr_scheduler $lr_scheduler \
                                                                                --w_regression $w_regression \
                                                                                --w_classify $w_classify \
                                                                                --data_path "$DATA_PATH" \
                                                                                --output_dir "models/$EXP_NAME" \
                                                                                --log_steps 5 \
                                                                                --log_wandb \
                                                                                --project_name "$PROJECT_NAME" \
                                                                                --experiment_name "$EXP_NAME"
                                                                        done
                                                            elif ["$task" == "MultitaskOrthogonal"]
                                                            then
                                                                for w_classify in "${W_CLASSIFY[@]}"
                                                                    for w_regression in "${W_REGRESSION[@]}" 
                                                                        for w_grad in "${W_GRAD[@]}" 
                                                                            do
                                                                                EXP_NAME="$network-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv1_$hidden_size_conv1-hidden_size_conv2_$hidden_size_conv2-hidden_size_conv3_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-learning_rate_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify-w_grad_$w_grad" \
                                                                                python train.py \
                                                                                    --seed $seed \
                                                                                    --task $task\
                                                                                    --batch_size $batch_size \
                                                                                    --epochs $epochs \
                                                                                    --p_dropout $p_dropout \
                                                                                    --network LSTM \
                                                                                    --input_dim 3 \
                                                                                    --hidden_size_conv1 $hidden_size_conv1 \
                                                                                    --hidden_size_conv2 $hidden_size_conv2 \
                                                                                    --hidden_size_conv3 $hidden_size_conv3 \
                                                                                    --kernel_size $kernel_size \
                                                                                    --num_heads $num_heads \
                                                                                    --n_classes 12 \
                                                                                    --learning_rate $lr \
                                                                                    --lr_scheduler $lr_scheduler \
                                                                                    --w_regression $w_regression \
                                                                                    --w_classify $w_classify \
                                                                                    --w_grad $w_grad \
                                                                                    --data_path "$DATA_PATH" \
                                                                                    --output_dir "models/$EXP_NAME" \
                                                                                    --log_steps 5 \
                                                                                    --log_wandb \
                                                                                    --project_name "$PROJECT_NAME" \
                                                                                    --experiment_name "$EXP_NAME"
                                                                            done
                                                            elif ["$task" == "MultitaskOrthogonalTracenorm"]
                                                            then
                                                                for w_classify in "${W_CLASSIFY[@]}"
                                                                    for w_regression in "${W_REGRESSION[@]}" 
                                                                        for w_grad in "${W_GRAD[@]}" 
                                                                            for w_trace_norm in "${W_TRACENORM[@]}" 
                                                                                do
                                                                                    python train.py \
                                                                                    --seed $seed \
                                                                                    --task $task\
                                                                                    --batch_size $batch_size \
                                                                                    --epochs $epochs \
                                                                                    --p_dropout $p_dropout \
                                                                                    --network LSTM \
                                                                                    --input_dim 3 \
                                                                                    --hidden_size_conv1 $hidden_size_conv1 \
                                                                                    --hidden_size_conv2 $hidden_size_conv2 \
                                                                                    --hidden_size_conv3 $hidden_size_conv3 \
                                                                                    --kernel_size $kernel_size \
                                                                                    --num_heads $num_heads \
                                                                                    --n_classes 12 \
                                                                                    --learning_rate $lr \
                                                                                    --lr_scheduler $lr_scheduler \
                                                                                    --w_regression $w_regression \
                                                                                    --w_classify $w_classify \
                                                                                    --w_grad $w_grad \
                                                                                    --w_trace_norm $w_trace_norm \
                                                                                    --data_path "$DATA_PATH" \
                                                                                    --output_dir "models/$EXP_NAME" \
                                                                                    --log_steps 5 \
                                                                                    --log_wandb \
                                                                                    --project_name "$PROJECT_NAME" \
                                                                                    --experiment_name "$EXP_NAME"
                                                                                done
                                                            fi
                                                        done




                                

                                
                                ["$task" == "MultitaskOrthogonal"] || ["$task" == "MultitaskOrthogonalTracenorm"]]
                                for w_reg in "${W_REGRESSION[@]}"
                                    for w_cls in "${W_CLASSIFY[@]}"
                                        for w_grad in "${W_GRAD[@]}"
                                            for w_trace_norm in "${W_TRACENORM[@]}"

                                                for hidden_size_lstm1 in "${HIDDEN_SIZE_LSTM1[@]}"
                                                    for hidden_size_lstm2 in "${HIDDEN_SIZE_LSTM2[@]}"
                                                        do
                                                            echo "Running task: $task, using LSTM network"
                                                            EXP_NAME="LSTM-seed_$seed-task_$task-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-learning_rate_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_reg-w_classify_$w_cls-w_grad_$w_grad-w_trace_norm_$w_trace_norm \
                                                                python train.py \
                                                                    --seed $seed \
                                                                    --task $task\
                                                                    --batch_size $batch_size \
                                                                    --epochs $epochs \
                                                                    --p_dropout $p_dropout \
                                                                    --network LSTM \
                                                                    --input_dim 3 \
                                                                    --hidden_size_lstm1 $hidden_size_lstm1 \
                                                                    --hidden_size_lstm2 $hidden_size_lstm2 \
                                                                    --n_classes 12 \
                                                                    --learning_rate $lr \
                                                                    --lr_scheduler $lr_scheduler \
                                                                    --w_regression $w_reg\
                                                                    --w_classify $w_cls \
                                                                    --w_grad $w_grad \
                                                                    --w_trace_norm $w_trace_norm \
                                                                    --data_path "$DATA_PATH" \
                                                                    --output_dir "models/$EXP_NAME" \
                                                                    --log_steps 5 \
                                                                    --log_wandb \
                                                                    --project_name Test \
                                                                    --experiment_name "$EXP_NAME"
                                                        done

                                                            echo "Running task: $task, using CNN-Attention"
                                                            python train.py \
                                                                --task MultitaskOrthogonalTracenorm\
                                                                --network LSTM \
                                                                --batch_size 512 \
                                                                --epochs 20 \
                                                                --p_dropout 0.25 \
                                                                --input_dim 3 \
                                                                --hidden_size_lstm1 128 \
                                                                --hidden_size_lstm2 64 \
                                                                --n_classes 12 \
                                                                --learning_rate 0.001 \
                                                                --lr_scheduler CosineAnnealingWarmRestarts \
                                                                --w_regression 0.1 \
                                                                --w_classify 0.5 \
                                                                --w_grad 0.5 \
                                                                --w_trace_norm 0.001 \
                                                                --data_path "$DATA_PATH" \
                                                                --output_dir "models/CNN-Attention-$task" \
                                                                --log_steps 5 \
                                                                --log_wandb \
                                                                --project_name Test \
                                                                --experiment_name "$task-LSTM"
                                                        done
