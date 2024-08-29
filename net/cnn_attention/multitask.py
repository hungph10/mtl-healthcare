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

class CNNAttention1D(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_conv1,
        hidden_conv2,
        hidden_conv3,
        kernel_size,
        num_heads,
        dropout,
        num_classes,
        sequence_length
    ):
        super(CNNAttention1D, self).__init__()
        
        # Padding for keep original sequence length
        padding = int((kernel_size - 1) / 2)
        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_conv1, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_conv1, hidden_conv2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_conv2, hidden_conv3, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        # Attention layer
        self.attention = MultiheadSelfAttention(
            input_dim=hidden_conv3,
            hidden_dim=hidden_conv3,
            num_heads=num_heads,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_conv3)
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_conv3, 4 * hidden_conv3),
            nn.ReLU(),
            nn.Linear(4 * hidden_conv3, hidden_conv3),
            nn.ReLu(),
            nn.Dropout(p=dropout)
        )
        # Tasks layer
        self.cls = nn.Line

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        conv_output = self.conv(x.transpose(1, 2)).transpose(1,2)
        attention_output = self.attention(conv_output)
        norm_output = self.layer_norm(conv_output + attention_output)
        print(x.shape)
        

#         # CNN layers
#         x = self.relu(self.conv1(x))
#         print(x.shape)
#         x = self.relu(self.conv2(x))
#         print(x.shape)
        
#         x = self.relu(self.conv3(x))
#         print(x.shape)
        
#         # Prepare for attention
#         x = x.transpose(1, 2)  # (batch_size, sequence_length, 256)
#         # Attention layer
#         x = self.attention(x)
#         print("After attention: ", x.shape)
#         # # Final layers
#         x = self.relu(self.fc(x))
#         print(x.shape)
#         x = self.dropout(x)

#         cls_pred = self.cls_head(x)

#         x = torch.mean(x, dim=1)  # (batch_size, 256)
#         reg_pred = self.reg_head(x)
        
#         return cls_pred, reg_pred
# # Example usage
batch_size = 1
sequence_length = 20
input_dim = 3
num_classes = 5

model = CNNAttention1D(
    input_dim,
    hidden_conv1=64,
    hidden_conv2=128,
    hidden_conv3=256,
    kernel_size=3,
    num_classes=5,
    num_heads=16,
    dropout=0.25,
    sequence_length=sequence_length
)
input_tensor = torch.randn(batch_size, sequence_length, input_dim)
# print(model)
print(input_tensor.shape)
model(input_tensor)


# cls_pred, reg_pred = model(input_tensor)

# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {cls_pred.shape} - {reg_pred.shape}")
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")