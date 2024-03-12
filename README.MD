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
        --w_regression 0.33 \
        --w_classify 0.33 \
        --w_grad 0.33 \
        --data_path "data/multitask_cls12_regr.npz" \
        --output_dir "models/multitask_LSTM" \
        --project_name="Multitask healthcare" \
        --experiment_name="mtl-LSTM-128-64-orthogonal" \
        # --log_wandb

    ```
    
    Open terminal and run train.sh
    ```
    sh train.sh
    ```


