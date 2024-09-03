python train.py \
    --task MultitaskOrthogonalTracenorm\
    --network LSTM \
    --batch_size 512 \
    --epochs 600 \
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
    --data_path data/official/train_val_test-1508.npz \
    --output_dir models/test

# python train.py \
#     --task MultitaskOrthogonalTracenorm\
#     --network CNN-Attention \
#     --batch_size 512 \
#     --epochs 600 \
#     --p_dropout 0.25 \
#     --input_dim 3 \
#     --hidden_size_conv1 8 \
#     --hidden_size_conv2 16 \
#     --hidden_size_conv3 32 \
#     --kernel_size 3 \
#     --num_heads 2 \
#     --n_classes 12 \
#     --learning_rate 0.001 \
#     --lr_scheduler CosineAnnealingWarmRestarts \
#     --w_regression 0.1 \
#     --w_classify 0.5 \
#     --w_grad 0.5 \
#     --w_trace_norm 0.001 \
#     --data_path data/official/train_val_test-1508.npz \
#     --output_dir models/test