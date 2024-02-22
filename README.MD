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
        --data_path <Path to data file .npz> \
        --output_dir <Output dir model> \
        --project_name="Name project logged on wandb" \
        --experiment_name="Experiment name logged on wandb" \
        --log_wandb 
    ```


