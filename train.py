import argparse

from train_utils import (
    set_random_seed,
    get_trainer
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyper parameters for training")
    parser.add_argument('--seed', type=int, default=None, help='Set the random seed')
    parser.add_argument(
        '--task',
        type=str,
        default="Multitask",
        help='Training task, task should be one of these tasks: Classify, Regression, \
Multitask, MultitaskOrthogonal, MultitaskOrthogonalTracenorm'
    )
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--p_dropout', type=float, help='Dropout probability')

    # Network dimensions
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument("--network", type=str, help='Network architecture')
    # LSTM hyper parameters
    parser.add_argument('--hidden_size_lstm1', type=int, help='Number of hidden units in the first LSTM layer')
    parser.add_argument('--hidden_size_lstm2', type=int, help='Number of hidden units in the second LSTM layer')
    # CNN-Attention hyper parameters
    parser.add_argument('--hidden_size_conv1', type=int, help='Number of hidden units in the first Conv1D')
    parser.add_argument('--hidden_size_conv2', type=int, help='Number of hidden units in the second Conv1D')
    parser.add_argument('--hidden_size_conv3', type=int, help='Number of hidden units in the third Conv1D')
    parser.add_argument('--kernel_size', type=int, help='Kernel size of Conv1D')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads ')
    
    parser.add_argument('--n_classes', type=int, help='Number of output classes')
    
    # Learning rate scheduler
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default=None,
        help='Default is constant learning rate, support these learning rate scheduler: \
StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help='Gamma value for LR scheduler (StepLR and ExponentialLR)'
    )
    parser.add_argument(
        '--step_size', type=int, default=100,
        help='Step size reduce learning rate for StepLR scheduler'
    )
    parser.add_argument(
        '--T_max', type=int, default=200,
        help='Max iterations for CosineAnnealingLR'
    )
    parser.add_argument(
        '--T_0', type=int, default=100,
        help='Number of iterations until the first restart for CosineAnnealingWarmRestarts'
    )
    parser.add_argument(
        '--T_mul', type=int, default=2,
        help='A factor by which T_i increases after a restart for CosineAnnealingWarmRestarts'
    )

    # Weight for aggregated loss
    parser.add_argument('--w_regression', type=float, default=1, help='Weight regression loss')
    parser.add_argument('--w_classify', type=float, default=1, help='Weight classify loss')
    parser.add_argument('--w_grad', type=float, default=1, help='Weight orthogonal gradient loss')
    parser.add_argument('--w_trace_norm', type=float, default=1, help='Weight tracenorm loss')

    # Location of data and checkpoint 
    parser.add_argument('--data_path', type=str, help='Path to the data training')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving models')

    # WandB / Console logging
    parser.add_argument('--log_steps', type=int, help='Logging steps during training')
    parser.add_argument('--log_console', action='store_true', help='Enable console logging')
    parser.add_argument('--log_wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--project_name', type=str, default='Project demo', help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default='Experiment demo', help='WandB experiment name')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.seed:
        print("Initialize parameters with random seed: ", args.seed)
        set_random_seed(seed=args.seed)
    else:
        print("Initialize parameters random without seed")

    trainer = get_trainer(args=args)
    trainer.train()
