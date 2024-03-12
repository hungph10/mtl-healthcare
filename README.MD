# Multitask learning: Respiration Estimation & Posture classification

_Authors: **Pham Van Hung**_

This repository contains code for training, experiments multitask classification and regression.

## Setup Instructions

1. **Download data File**: 
   - You need to download the data file for training and testing.
   - After downloaded, place the data file in the `data/` folder within this repository.

2. **API Token**: 
    - Create a file named `.env` and insert your Wandb API token for tracking log:
        ```
        WANDB_API_KEY=<Your API token>
        ```

3. **Training**: 
   - Firstly, install dependencies:
   ```
   pip install -r requirements.txt
   ```
   - Create file `train.sh`, and insert script training
    ```
    # Training multitask orthogonal
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

    ```
    
    Open terminal and run train.sh
    ```
    sh train.sh
    ```


