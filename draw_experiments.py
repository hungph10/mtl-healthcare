import argparse
import random

import numpy as np
import torch

from dataset import get_data_mtl
from dataset import MultitaskDataset
from trainer.multitask_orthogonal_tracenorm_trainer import MultitaskOrthogonalTracenormTrainer
from net import (
    MultitaskLSTM,
    cls_metric,
    cls_loss_fn,
    reg_loss_fn,
    reg_metric
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyper parameters for training")
    # Hyper parameter for training
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--n_hidden_1', type=int, help='Number of hidden units in the LSTM layer')
    parser.add_argument('--n_hidden_2', type=int, help='Number of hidden units in the LSTM layer')
    parser.add_argument('--n_classes', type=int, help='Number of output classes')
    parser.add_argument('--p_dropout', type=float, help='Dropout probability')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Set the random seed')
    parser.add_argument('--log_steps', type=int, help='Logging steps during training')
    parser.add_argument('--w_regression', type=float, default=1, help='Weight regression loss')
    parser.add_argument('--w_classify', type=float, default=1, help='Weight classify loss')
    parser.add_argument('--w_grad', type=float, default=1, help='Weight gradient loss')
    parser.add_argument('--w_trace_norm', type=float, default=1, help='Weight gradient loss')
    parser.add_argument('--data_path', type=str, help='Path to the data training')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving models')
    parser.add_argument('--log_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='Project demo', help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default='Experiment demo', help='WandB experiment name')
    
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


class Args:
    batch_size = 512
    epochs = 600
    input_dim = 3
    n_hidden_1 = 128
    n_hidden_2 = 64
    n_classes = 12
    p_dropout = 0.25
    learning_rate = 0.001
    seed = 42
    log_console=False
    log_steps = 5
    w_regression = 0.4
    w_classify = 0.9
    w_grad = 0.3
    w_trace_norm = 0.001
    data_path = "data/train_validate.npz"
    output_dir = "models/test1"
    log_wandb = False
    project_name = ""
    experiment_name = ""

args = Args()

# Set the random seed
if args.seed:
    print("Initialize parameters with random seed: ", args.seed)
    set_random_seed(seed=args.seed)
else:
    print("Initialize parameters random without seed")

# Load data
data = np.load(args.data_path)
print("Loading data from {}...".format(args.data_path))
tensor_data = get_data_mtl(data=data)
train_dataset = MultitaskDataset(
    features=tensor_data["x_train"],
    cls_target=tensor_data["y_train_cls"],
    reg_target=tensor_data["y_train_reg"]
)
test_dataset = MultitaskDataset(
    features=tensor_data["x_test"],
    cls_target=tensor_data["y_test_cls"],
    reg_target=tensor_data["y_test_reg"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    # Initialize the PyTorch model
    model = MultitaskLSTM(
        input_size=args.input_dim,
        hidden_size_1=args.n_hidden_1,
        hidden_size_2=args.n_hidden_2,
        output_size=args.n_classes,
        dropout=args.p_dropout
    )

    model.load_state_dict(
        state_dict=torch.load(
            f=model_path,
            map_location=torch.device("cpu")
        )
    )
    model = model.to(device)
    return model

def make_trainer(model):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate
    )
    return  MultitaskOrthogonalTracenormTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        cls_loss_fn=cls_loss_fn,
        reg_loss_fn=reg_loss_fn,
        cls_metric=cls_metric,
        reg_metric=reg_metric,
        optimizer=optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        log_steps=args.log_steps,
        log_console=args.log_console,
        log_wandb=args.log_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        weight_regression=args.w_regression,
        weight_classify=args.w_classify,
        weight_grad=args.w_grad,
        weight_trace_norm=args.w_trace_norm
    )


def evaluate(model_path):
    model = load_model(model_path=model_path)
    trainer = make_trainer(model=model)
    return trainer.evaluate(
        test_dataloader=trainer.test_dataloader,
        model=model,
        compute_cls_loss=cls_loss_fn,
        compute_reg_loss=reg_loss_fn,
        cls_metric=cls_metric,
        reg_metric=reg_metric
    )



MTT_TRACENORM = "models/cuong/0.4-0.9-0.3-0.001/best_cls.pth"
model_tracenorm = load_model(model_path=MTT_TRACENORM)
evaluate(MTT_TRACENORM)


MTT_ORTHGONAL_PATH = "models/old_experiments/models/w_regression-0.1/w_classify-0.5/w_grad-0.5/LSTM_n_hidden1-_n_hidden2-_p_dropout-_-learning_rate_w_regression-0.1_w_classify-0.5_w_grad-0.5/best_cls.pth" 
evaluate(MTT_ORTHGONAL_PATH)



from tqdm import trange
X = []
Y = []
for i in trange(len(test_dataset)):
    idx = np.random.randint(0, 200, size=200)
    x, y_cls, y_reg = test_dataset[i]
    x = x[idx]
    y_cls = y_cls[idx]
    Y.extend(y_cls.tolist())
    out_lstm, _ = model_tracenorm.lstm(x)
    X.append(out_lstm)

X = torch.concatenate(X)
X = X.detach().numpy()



import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px


tsne_model = TSNE(n_components=2, perplexity=5)
tsne_data = tsne_model.fit_transform(X)
color_map = {
    0: "blue",
    1: "green",
    2: "red",
    3: "purple",
    4: "orange",
    5: "teal",
    6: "fuchsia",
    7: "gray",
    8: "olive",
    9: "maroon",
    10: "navy",
    11: "lime"
}
# Create scatter plot with color based on target variable
fig = px.scatter(
    x=tsne_data[:, 0],
    y=tsne_data[:, 1],
    color=Y,  # Color based on target variable
    opacity=0.7,  # Set opacity for better visibility
    width=2200,
    height=1400,
    color_discrete_sequence=color_map,
    template="plotly_white"
)

# Update layout
fig.update_layout(
    title="t-SNE Visualization",
    xaxis_title="",
    yaxis_title="",
    legend_title="Target",  # Add legend title
)

fig.show()