# Multitask learning: Respiration Estimation & Posture classification

_Authors: **Vu Hoang Dieu, Pham Van Hung, Nghia DV, Dat, Cuong**_

This repository contains code for training, experiments multitask classification and regression.

## Setup Instructions

1. **Download data File**: 
   - You need to download the data file for training and testing.
   - After downloaded, place the data file in the `data/` folder within this repository.

2. **API Token**: 
    - Create a file named `.env` and insert your Wandb API token for tracking log. To get API token, \
access *https://wandb.ai/authorize* (login via Google account):
        ```
        WANDB_API_KEY=<Your API token>
        ```

3. **Training**: 
   
    *Ensure you installed* **python** *on your machine.*
    - You should create venv:

        ```
        python -m venv venv
        source venv/bin/activate
        (venv) pip install -U pip
        ```


    - Firstly, install dependencies:
        ```
        (venv) pip install -r requirements.txt
        ```
    
    #### Machine learning experiments:
    - Classify task:
        ```
        python train.py \
            --data_path <Data path to npz file> \
            --output_report_csv <Path to csv output file>
    
    #### Neural network experiments
    - Usage
        ```
        usage: train.py [-h] 


        options:
        -h, --help            show this help message and exit
        --seed SEED           Set the random seed
        --task TASK           Training task, task should be one of these tasks: Classify, Regression, Multitask, MultitaskOrthogonal, MultitaskOrthogonalTracenorm
        --batch_size BATCH_SIZE
                            Batch size for training
        --epochs EPOCHS       Number of training epochs
        --p_dropout P_DROPOUT
                            Dropout probability
        --input_dim INPUT_DIM
                            Input dimension
        --network NETWORK     Network architecture, support only: LSTM, CNN-Attention
        --hidden_size_lstm1 HIDDEN_SIZE_LSTM1
                            Number of hidden units in the first LSTM layer (LSTM architecture)
        --hidden_size_lstm2 HIDDEN_SIZE_LSTM2
                            Number of hidden units in the second LSTM layer (LSTM architecture)
        --hidden_size_conv1 HIDDEN_SIZE_CONV1
                            Number of hidden units in the first Conv1D (CNN-Attention architecture)
        --hidden_size_conv2 HIDDEN_SIZE_CONV2
                            Number of hidden units in the second Conv1D (CNN-Attention architecture)
        --hidden_size_conv3 HIDDEN_SIZE_CONV3
                            Number of hidden units in the third Conv1D (CNN-Attention architecture)
        --kernel_size KERNEL_SIZE
                            Kernel size of Conv1D (CNN-Attention architecture)
        --num_heads NUM_HEADS
                            Number of attention heads (CNN-Attention architecture)
        --n_classes N_CLASSES
                            Number of output classes
        --learning_rate LEARNING_RATE
                            Learning rate
        --lr_scheduler LR_SCHEDULER
                            Default is constant learning rate, support these learning rate scheduler: StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
        --gamma GAMMA         Gamma value for LR scheduler (StepLR and ExponentialLR)
        --step_size STEP_SIZE
                            Step size reduce learning rate for StepLR scheduler
        --T_max T_MAX         Max iterations for CosineAnnealingLR
        --T_0 T_0             Number of iterations until the first restart for CosineAnnealingWarmRestarts
        --T_mul T_MUL         A factor by which T_i increases after a restart for CosineAnnealingWarmRestarts
        --w_regression W_REGRESSION
                            Weight regression loss
        --w_classify W_CLASSIFY
                            Weight classify loss
        --w_grad W_GRAD       Weight orthogonal gradient loss
        --w_trace_norm W_TRACE_NORM
                            Weight tracenorm loss
        --data_path DATA_PATH
                            Path to the data training
        --output_dir OUTPUT_DIR
                            Output directory for saving models
        --log_steps LOG_STEPS
                            Logging steps during training
        --log_console         Enable console logging
        --log_wandb           Enable Wandb logging
        --project_name PROJECT_NAME
                            Wandb project name
        --experiment_name EXPERIMENT_NAME
                            Wandb experiment name
        ```
   - Create file `train.sh`, and insert script training, without logging during training (still display tqdm progress bar, recommend for training on server without internet connection)
        ```
        # Example training multitask orthogonal using LSTM
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
            --data_path <Data dir>/train_val_test-1508.npz \>
            --output_dir models/LSTM-multitask-orthogonal-tracenorm
            
        ```
    - If logging to console:
        ```
        python train.py \
            ... 
            --log_steps 5 \
            --log_console
        ```
    - If logging to wandb:
        ```
        python train.py \
            ... 
            --log_steps 5 \
            --log_wandb \
            --project_name Accelerometer Signal 
            --experiment_name Multitask-Orthogonal-Tracenorm-LSTM
        ```
    
    - Open terminal and run train.sh
        ```
        (venv) bash train.sh
        ```
