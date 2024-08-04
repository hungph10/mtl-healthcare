import torch
from torch import nn  




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultitaskMLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                    num_lstm_out=128, num_lstm_layers=1, 
                    conv1_nf=32, conv2_nf=64, conv3_nf=32,
                    lstm_drop_p=0.7, fc_drop_p=0.65):
        super(MultitaskMLSTMfcn, self).__init__()

        # Common attributes
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        # LSTM configuration
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        # Convolutional layers configuration
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        # Dropout configuration
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_features, 
                            hidden_size=num_lstm_out,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(num_features, conv1_nf, 8)
        self.conv2 = nn.Conv1d(conv1_nf, conv2_nf, 5)
        self.conv3 = nn.Conv1d(conv2_nf, conv3_nf, 3)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(conv1_nf)
        self.bn2 = nn.BatchNorm1d(conv2_nf)
        self.bn3 = nn.BatchNorm1d(conv3_nf)

        # Squeeze-and-Excitation layers
        self.se1 = SELayer(conv1_nf)
        self.se2 = SELayer(conv2_nf)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_drop_p)

        # Fully connected layers for each task
        self.fc_reg = nn.Linear(conv3_nf + num_lstm_out, 1)  # Regression
        self.fc_cls = nn.Linear(conv3_nf + num_lstm_out, num_classes)  # Classification

    def forward(self, x, seq_lens):
        # Pack the sequence for LSTM processing
        x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x_packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=0.0)
        lstm_out_last = lstm_out[:, -1, :]

        # Convolutional layers processing
        x = x.transpose(2, 1)  # Prepare for Conv1d
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.se1(x)
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = torch.mean(x, 2)  # Global Average Pooling

        # Concatenate LSTM and CNN features
        x_all = torch.cat((lstm_out_last, x), dim=1)

        # Classification output
        cls_out = self.fc_cls(x_all)
        # Undo the softmax since it is contained in CrossEntropyLoss
        # cls_out = F.log_softmax(cls_out, dim=1) 

        # Regression output
        reg_out = self.fc_reg(x_all)
        reg_out = reg_out.squeeze(-1)  # Remove last dimension to match the required output shape

        return cls_out, reg_out
