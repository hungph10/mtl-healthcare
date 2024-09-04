
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

# Network supported
NETWORKS=("LSTM" "CNN-Attention")

# Hyper parameters for LSTM architecture
HIDDEN_SIZE_LSTM1=(128)
HIDDEN_SIZE_LSTM2=(64)

# Hyper parameters for CNN-Attention architecture
HIDDEN_SIZE_CONV1=(64)
HIDDEN_SIZE_CONV2=(128)
HIDDEN_SIZE_CONV3=(256)
KERNEL_SIZE=(3)
NUM_HEADS=(16)
