import torch.nn as nn


class MultitaskMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(MultitaskMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        cls_out = self.cls(x) 
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        reg_out = self.reg(x).flatten() 
        
        return reg_out, cls_out
