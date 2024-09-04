# python train.py \
#     --task MultitaskOrthogonalTracenorm\
#     --network LSTM \
#     --batch_size 512 \
#     --epochs 600 \
#     --p_dropout 0.25 \
#     --input_dim 3 \
#     --hidden_size_lstm1 128 \
#     --hidden_size_lstm2 64 \
#     --n_classes 12 \
#     --learning_rate 0.001 \
#     --lr_scheduler CosineAnnealingWarmRestarts \
#     --w_regression 0.1 \
#     --w_classify 0.5 \
#     --w_grad 0.5 \
#     --w_trace_norm 0.001 \
#     --data_path data/official/train_val_test-1508.npz \
#     --output_dir models/test

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

