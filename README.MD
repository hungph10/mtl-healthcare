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
   
   Ensure you installed python on your machine.
   You should create venv:
   ```
   python -m venv venv
   source venv/bin/activate
   (venv) pip install -U pip
   ```
   - Firstly, install dependencies:
   ```
   (venv) pip install -r requirements.txt
   ```
   - Create file `train.sh`, and insert script training
    ```
    # Training multitask orthogonal
    DATA_PATH=<Path to .npz data>
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



    ```
    
    Open terminal and run train.sh
    ```
    (venv) bash train.sh
    ```

4. **Hyper parameters**

    Open terminal and run this command to read parameter description:
    ```
    (venv) python train.py -h
    ```