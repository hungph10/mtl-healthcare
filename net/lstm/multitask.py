from torch import nn


class MultitaskLSTM(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(MultitaskLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):

        """
        x: (512, 300, 3)
        
        """
        out, _ = self.lstm(x)
        
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, out.shape[2])
        cls_out = self.cls(cls_out)

        out, _ = self.lstm2(out[:, -1, :].view(x.size(0), 1, -1))
        out = self.dropout(out)
        reg_out = self.reg(out[:, -1, :]).flatten()

        return reg_out, cls_out
