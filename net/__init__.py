import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from .lstm import ClassifyLSTM, RegressionLSTM, MultitaskLSTM
from .cnn_attention import ClassifyCNNAttention1D, RegressionCNNAttention1D, MultitaskCNNAttention1D



from .kan_multitask_lstm import KANMultitaskLSTM


cls_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.MSELoss()

reg_metric = nn.L1Loss()
def cls_metric(logit, label):
    pred_label = torch.argmax(logit, dim=-1)
    acc = accuracy_score(pred_label.cpu(), label.cpu())
    f1 = f1_score(pred_label.cpu(), label.view(-1).cpu(), average="macro")
    return acc, f1