import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout):
        super(MultiheadSelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        
        return torch.matmul(attention, V)


class ClassifyCNNAttention1D(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size_conv1,
        hidden_size_conv2,
        hidden_size_conv3,
        kernel_size,
        num_heads,
        dropout,
        num_classes
    ):
        super(ClassifyCNNAttention1D, self).__init__()
        
        # Padding for keep original sequence length
        padding = int((kernel_size - 1) / 2)
        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size_conv1, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_size_conv1, hidden_size_conv2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_size_conv2, hidden_size_conv3, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        # Attention layer
        self.attention = MultiheadSelfAttention(
            input_dim=hidden_size_conv3,
            hidden_dim=hidden_size_conv3,
            num_heads=num_heads,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size_conv3)
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size_conv3, 4 * hidden_size_conv3),
            nn.ReLU(),
            nn.Linear(4 * hidden_size_conv3, hidden_size_conv3),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        # Tasks layer
        self.cls = nn.Linear(hidden_size_conv3, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        conv_output = self.conv(x.transpose(1, 2)).transpose(1,2)
        
        attention_output = self.attention(conv_output)
        norm_output = self.layer_norm(conv_output + attention_output)
        
        mlp_output = self.mlp(norm_output)
        norm_mlp_output = self.layer_norm(norm_output + mlp_output)
        
        cls_output = self.cls(norm_mlp_output)
        cls_output = cls_output.view(-1, cls_output.shape[2])
        return cls_output