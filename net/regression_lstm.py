from torch import nn


class RegressionLSTM(nn.Module):
    
    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            dropout,
            output_size=1
        ):
        super(RegressionLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Linear(hidden_size_2, output_size)

        
    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out[:, -1, :].view(x.size(0), 1, -1))
        out = self.dropout(out)
        reg_out = self.reg(out[:, -1, :]).flatten()

        return reg_out