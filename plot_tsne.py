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
    data_path = "data/multitask_cls12_regr.npz"
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
MTT_TRACENORM = "models/paper_experiments/multitask_LSTM/0.4-0.9-0.3-0.001/best_cls.pth"
model_tracenorm = load_model(model_path=MTT_TRACENORM)



from tqdm import trange
X = []
Y = []
for i in trange(len(test_dataset)):
    idx = np.random.randint(0, 200, size=200)
    x, y_cls, y_reg = test_dataset[i]
    x = x[idx]
    y_cls = y_cls[idx]
    Y.extend(y_cls.tolist())
    X.append(x)

Y = [str(_) for _ in Y]


import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

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

def plot_tSNE(
        model,
        title = "t-SNE Visualization"
    ):
    features = [model.lstm(x)[0] for x in X]
    features = torch.concatenate(features)
    features = features.detach().numpy()
    tsne_model = TSNE(n_components=2, perplexity=5)
    tsne_data = tsne_model.fit_transform(features)
    # Create scatter plot with color based on target variable
    fig = px.scatter(
        x=tsne_data[:, 0],
        y=tsne_data[:, 1],
        color=Y,  # Color based on target variable
        opacity=0.7,  # Set opacity for better visibility
        color_discrete_sequence=color_map,
        template="plotly_white"
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        legend_title="Posture",  # Add legend title
        width=600,
        height=600
        )
    fig.show()



plot_tSNE(
    model=model_tracenorm,
    title="Multitask learning + Orthogonal Gradient + Tracenorm"
)



from tqdm import trange
X = []
Y = []
for i in trange(len(test_dataset)):
    idx = np.random.randint(0, 200, size=200)
    x, y_cls, y_reg = test_dataset[i]
    x = x[idx]
    y_cls = y_cls[idx]
    Y.extend(y_cls.tolist())
    X.append(x)

Y = [str(_) for _ in Y]


import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

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

def plot_tSNE(
        model,
        title = "t-SNE Visualization"
    ):
    features = [model.lstm(x)[0] for x in X]
    features = torch.concatenate(features)
    features = features.detach().numpy()
    tsne_model = TSNE(n_components=2, perplexity=5)
    tsne_data = tsne_model.fit_transform(features)
    # Create scatter plot with color based on target variable
    fig = px.scatter(
        x=tsne_data[:, 0],
        y=tsne_data[:, 1],
        color=Y,  # Color based on target variable
        opacity=0.7,  # Set opacity for better visibility
        color_discrete_sequence=color_map,
        template="plotly_white"
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        legend_title="Posture",  # Add legend title
        width=600,
        height=600
        )
    fig.show()


MTT_TRACENORM = "models/paper_experiments/multitask_LSTM/0.4-0.9-0.3-0.001/best_cls.pth"
model_tracenorm = load_model(model_path=MTT_TRACENORM)



plot_tSNE(
    model=model_tracenorm,
    title="Multitask learning + Orthogonal Gradient + Tracenorm"
)



MTT_ORTHOGONAL = "models/paper_experiments/multitask_LSTM/0.4-0.9-0.3-0.001/best_cls.pth"
model_orthogonal = load_model(model_path=MTT_ORTHOGONAL)
plot_tSNE(
    model=model_orthogonal,
    title="Multitask learning + Orthogonal Gradient"
)