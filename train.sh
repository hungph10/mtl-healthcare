
source config.sh
# Loop through each task and run the training script
for seed in "${SEEDS[@]}"; do
    for batch_size in "${BATCH_SIZE[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            for p_dropout in "${DROPOUT[@]}"; do
                for lr in "${LEARNING_RATE[@]}"; do
                    for lr_scheduler in "${LR_SCHEDULER[@]}"; do
                        for network in "${NETWORKS[@]}"; do
                            if [ "$network" = "LSTM" ]
                            then
                                for hidden_size_lstm1 in "${HIDDEN_SIZE_LSTM1[@]}"; do
                                    for hidden_size_lstm2 in "${HIDDEN_SIZE_LSTM2[@]}"; do
                                        for task in "${TASKS[@]}"; do
                                            if [ "$task" = "Classify" ] || [ "$task" = "Regression" ]
                                            then
                                                short_name_task=$task
                                                EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-lr_$lr-lr_scheduler_$lr_scheduler" 
                                                echo "Running: $EXP_NAME"
                                                python train.py \
                                                    --seed $seed \
                                                    --task $task\
                                                    --batch_size $batch_size \
                                                    --epochs $epochs \
                                                    --p_dropout $p_dropout \
                                                    --network $network \
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
                                            elif [ "$task" = "Multitask" ]
                                            then
                                                short_name_task="MTT"
                                                for w_classify in "${W_CLASSIFY[@]}"; do
                                                    for w_regression in "${W_REGRESSION[@]}"; do 
                                                        EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-lr_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify" 
                                                        echo "Running: $EXP_NAME"
                                                        python train.py \
                                                            --seed $seed \
                                                            --task $task\
                                                            --batch_size $batch_size \
                                                            --epochs $epochs \
                                                            --p_dropout $p_dropout \
                                                            --network $network \
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
                                                done
                                            elif [ "$task" = "MultitaskOrthogonal" ]
                                            then
                                                short_name_task="MTT-Orthogonal"
                                                for w_classify in "${W_CLASSIFY[@]}"; do
                                                    for w_regression in "${W_REGRESSION[@]}"; do 
                                                        for w_grad in "${W_GRAD[@]}"; do
                                                            EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-lr_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify-w_grad_$w_grad" 
                                                            echo "Running: $EXP_NAME"
                                                            python train.py \
                                                                --seed $seed \
                                                                --task $task\
                                                                --batch_size $batch_size \
                                                                --epochs $epochs \
                                                                --p_dropout $p_dropout \
                                                                --network $network \
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
                                                    done
                                                done
                                            elif [ "$task" = "MultitaskOrthogonalTracenorm" ]
                                            then
                                                short_name_task="MTT-Orthogonal-Tracenorm"
                                                for w_classify in "${W_CLASSIFY[@]}"; do
                                                    for w_regression in "${W_REGRESSION[@]}"; do 
                                                        for w_grad in "${W_GRAD[@]}"; do 
                                                            for w_trace_norm in "${W_TRACENORM[@]}"; do
                                                                EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_lstm1_$hidden_size_lstm1-hidden_size_lstm2_$hidden_size_lstm2-lr_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify-w_grad_$w_grad-w_trace_norm_$w_trace_norm" 
                                                                echo "Running $EXP_NAME"
                                                                python train.py \
                                                                --seed $seed \
                                                                --task $task\
                                                                --batch_size $batch_size \
                                                                --epochs $epochs \
                                                                --p_dropout $p_dropout \
                                                                --network $network \
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
                                                        done
                                                    done
                                                done
                                            fi
                                        done
                                    done
                                done
                            elif [ "$network" = "CNN-Attention" ]
                            then
                                for hidden_size_conv1 in "${HIDDEN_SIZE_CONV1[@]}"; do
                                    for hidden_size_conv2 in "${HIDDEN_SIZE_CONV2[@]}"; do
                                        for hidden_size_conv3 in "${HIDDEN_SIZE_CONV3[@]}"; do
                                            for kernel_size in "${KERNEL_SIZE[@]}"; do
                                                for num_heads in "${NUM_HEADS[@]}"; do
                                                    for task in "${TASKS[@]}"; do
                                                        if [ "$task" = "Classify" ] || [ "$task" = "Regression" ]
                                                        then
                                                            short_name_task=$task
                                                            EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv_$hidden_size_conv1_$hidden_size_conv2_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-lr_$lr-lr_scheduler_$lr_scheduler" 
                                                            echo "Running: $EXP_NAME"
                                                            python train.py \
                                                                --seed $seed \
                                                                --task $task\
                                                                --batch_size $batch_size \
                                                                --epochs $epochs \
                                                                --p_dropout $p_dropout \
                                                                --network $network \
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
                                                        elif [ "$task" = "Multitask" ]
                                                        then
                                                            short_name_task="MTT"
                                                            for w_classify in "${W_CLASSIFY[@]}"; do
                                                                for w_regression in "${W_REGRESSION[@]}"; do 
                                                                    EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv_$hidden_size_conv1_$hidden_size_conv2_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-lr_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify" 
                                                                    echo "Running: $EXP_NAME"
                                                                    python train.py \
                                                                        --seed $seed \
                                                                        --task $task\
                                                                        --batch_size $batch_size \
                                                                        --epochs $epochs \
                                                                        --p_dropout $p_dropout \
                                                                        --network $network \
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
                                                            done
                                                        elif [ "$task" = "MultitaskOrthogonal" ]
                                                        then
                                                            short_name_task="MTT-Orthogonal"
                                                            for w_classify in "${W_CLASSIFY[@]}"; do
                                                                for w_regression in "${W_REGRESSION[@]}"; do 
                                                                    for w_grad in "${W_GRAD[@]}"; do 
                                                                        EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv_$hidden_size_conv1_$hidden_size_conv2_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-lr_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify-w_grad_$w_grad" 
                                                                        echo "Running: $EXP_NAME"
                                                                        python train.py \
                                                                            --seed $seed \
                                                                            --task $task\
                                                                            --batch_size $batch_size \
                                                                            --epochs $epochs \
                                                                            --p_dropout $p_dropout \
                                                                            --network $network \
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
                                                                done
                                                            done
                                                        elif [ "$task" = "MultitaskOrthogonalTracenorm" ]
                                                        then
                                                            short_name_task="MTT-Orthogonal-Tracenorm"
                                                            for w_classify in "${W_CLASSIFY[@]}"; do
                                                                for w_regression in "${W_REGRESSION[@]}"; do 
                                                                    for w_grad in "${W_GRAD[@]}"; do 
                                                                        for w_trace_norm in "${W_TRACENORM[@]}"; do 
                                                                            EXP_NAME="$short_name_task-$network-seed_$seed-batch_size_$batch_size-epochs_$epochs-p_dropout_$p_dropout-hidden_size_conv_$hidden_size_conv1_$hidden_size_conv2_$hidden_size_conv3-kernel_size_$kernel_size-num_heads_$num_heads-lr_$lr-lr_scheduler_$lr_scheduler-w_regression_$w_regression-w_classify_$w_classify-w_grad_$w_grad-w_trace_norm_$w_trace_norm" 
                                                                            echo "Running: $EXP_NAME"
                                                                            python train.py \
                                                                                --seed $seed \
                                                                                --task $task\
                                                                                --batch_size $batch_size \
                                                                                --epochs $epochs \
                                                                                --p_dropout $p_dropout \
                                                                                --network $network \
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
                                                                    done
                                                                done
                                                            done
                                                        fi
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
done

