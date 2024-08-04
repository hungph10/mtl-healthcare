from torch import nn



class MultitaskRNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(MultitaskRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, cls_out.size(-1)) 
        cls_out = self.cls(cls_out)

        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        reg_out = self.reg(out).squeeze()  

        return reg_out, cls_out
    