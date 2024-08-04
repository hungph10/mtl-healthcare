from torch import nn  



class MultitaskGRU(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(MultitaskGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size_1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size_1, hidden_size_2, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        cls_out = out[:, -1, :]  #
        cls_out = self.cls(cls_out)

        out, _ = self.gru2(out)
        out = self.dropout(out)
        reg_out = out[:, -1, :]  
        reg_out = self.reg(reg_out).squeeze()
        
        return reg_out, cls_out