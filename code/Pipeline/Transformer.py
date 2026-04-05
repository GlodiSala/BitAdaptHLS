import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=-1)
        swish = x_a * torch.sigmoid(x_a)
        return swish * x_b

class TransformerBlock(nn.Module):
    def __init__(self, input_dim=128, num_heads=4, hidden_dim=4096, dropout=0):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        
        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + self.dropout(attn_output))  # Add & norm

        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & norm

        return x

class StackedTransformer(nn.Module):
    def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=64, num_heads=2, hidden_dim=512, dropout=0, num_layers=2):
    # def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=128, num_heads=2, hidden_dim=1024, dropout=0.05, num_layers=4):
    # def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=64, num_heads=4, hidden_dim=512, dropout=0, num_layers=4):
    # def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=128, num_heads=4, hidden_dim=1024, dropout=0, num_layers=4):
    # def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=128, num_heads=4, hidden_dim=2048, dropout=0, num_layers=6):
    # def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=256, num_heads=8, hidden_dim=2048, dropout=0, num_layers=6):
    # def __init__(self, seq_length=4, token_dim=128, context_dim=2, embedding_dim=256, num_heads=8, hidden_dim=4096, dropout=0, num_layers=8):
        super(StackedTransformer, self).__init__()
        
        # Linear layer to embed the token dimension to a higher dimensional space
        self.embedding = nn.Linear(token_dim, embedding_dim)
    
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(input_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # for loading legacy models

        self.output = nn.Linear(embedding_dim, token_dim)

    def forward(self, x, idx=0):
        # Initial shape: (batch_size, sequence_length, token_dim)
        output_layer = self.output
        size_input = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(size_input[0], size_input[2], size_input[3] * size_input[1])
                    
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim))
        
        # Transpose for multi-head attention to (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        
        # Forward pass through each transformer block
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        # Transpose back to (batch_size, sequence_length, embedding_dim)
        x = x.permute(1, 0, 2)

        # Compute real and imaginary parts of the precoding weights
        csi_features = x

        outputs = output_layer(csi_features)
        
        # Split into real and imaginary parts and convert to complex
        outputs_r, outputs_i = outputs.chunk(2, dim=-1)
        precoding_matrix = torch.complex(outputs_r, outputs_i)
        precoding_matrix = precoding_matrix.permute(0, 2, 1)  # [batch, seq, token//2] -> [batch, token//2, seq]
        
        return precoding_matrix