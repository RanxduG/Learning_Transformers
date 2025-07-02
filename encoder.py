import torch
import math
from torch import nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute the scaled dot-product attention.

    query vector means: "What am I looking for?"
    key vector means: "What do I have to offer?"
    value vector means: "What I actually offer?"
    
    mask is used to make sure the decoder does not attend to future tokens during training.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k).
        key: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k).
        value: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v).
        mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k).
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len_q, d_v).
    """
    # query, key, value = 30 x 8 x 200 x 64
    d_k = query.size(-1) # 64
    scaled = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # 30 x 8 x 200 x 200
    
    if mask is not None:
        scaled += mask
    
    attn_weights = F.softmax(scaled, dim=-1) # 30 x 8 x 200 x 200
    output = torch.matmul(attn_weights, value) # 30 x 8 x 200 x 64
    
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Args:
        d_model: Dimension of the model.
        num_heads: Number of attention heads.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.head_dim = d_model // num_heads # 64
        self.num_heads = num_heads # 8
        self.qkv_layer = nn.Linear(d_model, d_model * 3) # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 x 512
    
    def forward(self, x, mask=None):
        batch_size, sequance_length, d_model = x.size() # 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequance_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        query, key, value = qkv.chunk(3, dim=-1) # each are 30 x 8 x 200 x 64
        values, attention = scaled_dot_product_attention(query, key, value, mask) # 30 x 8 x 200 x 64
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequance_length, self.d_model)
        # values = values.reshape(batch_size, sequance_length, self.d_model) # 30 x 200 x 512
        output = self.linear_layer(values) # 30 x 200 x 512
        return output
    

class LayerNormalization(nn.Module):
    """
    Layer normalization module.
    This module normalizes the input across the last dimension and applies learnable parameters gamma and beta.
    Args:
        parameter_shape: Shape of the learnable parameters (gamma and beta).
        eps: Small value to avoid division by zero.

    Returns:
        Normalized output tensor with the same shape as the input.

    The layer normalization is applied across the last dimension of the input tensor.
    The learnable parameters gamma and beta are used to scale and shift the normalized output.
    The output is computed as follows:
        output = gamma * (inputs - mean) / sqrt(variance + eps) + beta
    where mean and variance are computed across the last dimension of the input tensor.
    The mean and variance are computed as follows:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = ((inputs - mean) ** 2).mean(dim=-1, keepdim=True)
    The output tensor has the same shape as the input tensor.
    """
    
    def __init__(self, parameter_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.parameter_shape = parameter_shape
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))
    
    def forward(self, inputs):
        dims =[-(1 + 1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        variance = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        standardized = (variance + self.eps).sqrt()
        y = (inputs - mean) / standardized

        # Apply learnable parameters gamma and beta
        output = self.gamma * y + self.beta
        return output 
    
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Args:
        d_model: Dimension of the model.
        hidden: Dimension of the feed-forward layer.
        dropout: Dropout rate.
    """
    
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 x 2048
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): # 30 x 200 x 512
        x = self.linear1(x) # 30 x 200 x 2048
        x = self.relu(x) # 30 x 200 x 2048
        x = self.dropout(x)
        x = self.linear2(x) # 30 x 200 x 512
        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of multi-head attention and feed-forward network.
    Args:
        d_model: Dimension of the model.
        num_heads: Number of attention heads.
        dropout_prob: Dropout probability.

    output is a tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model, ffn_hidden, num_heads, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization([d_model])
        self.norm2 = LayerNormalization([d_model])

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout=dropout_prob)

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        residual = x
        x = self.attention(x, mask=None)
        x = self.dropout1(x)
        x = self.norm1(x + residual)  # Add & Norm
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)  # Add & Norm
        return x
        

class Encoder(nn.Module):
    """
    Encoder consisting of multiple encoder layers.
    
    Args:
        d_model: Dimension of the model.
        ffn_hidden: Dimension of the feed-forward layer.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout_prob: Dropout probability.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, num_layers, dropout_prob=0.1, max_seq_len=512):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.Sequential(*[
            EncoderLayer(d_model, ffn_hidden, num_heads, dropout_prob)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        print(f"Input to encoder: {x.shape}")
        x = self.pos_encoder(x)
        print(f"Encoded input: {x}")
        print(f"After positional encoding: {x.shape}")
        x = self.dropout(x)
        x = self.layers(x)
        print(f"Final encoder output: {x.shape}")
        print(f"Final encoder output: {x}")
        return x
