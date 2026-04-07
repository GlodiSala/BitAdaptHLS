import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, input_dim=64, num_heads=4, hidden_dim=512, dropout=0):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )

        # ReLU FFN — SwiGLU remplacé, pas de hidden_dim*2
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # RMSNorm — remplace LayerNorm
        self.norm1 = nn.RMSNorm(input_dim)
        self.norm2 = nn.RMSNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class StackedTransformer(nn.Module):
    def __init__(
        self,
        seq_length=4,
        token_dim=128,
        context_dim=2,
        embedding_dim=64,
        num_heads=4,
        hidden_dim=512,
        dropout=0,
        num_layers=4,
    ):
        super().__init__()

        self.embedding = nn.Linear(token_dim, embedding_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                input_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(embedding_dim, token_dim)

    def forward(self, x, idx=0):
        size_input = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(
            size_input[0], size_input[2], size_input[3] * size_input[1]
        )

        x = self.embedding(x)          # (batch, seq, embed_dim)
        x = x.permute(1, 0, 2)         # (seq, batch, embed_dim) pour MHA

        for layer in self.layers:
            x = layer(x)

        x = x.permute(1, 0, 2)         # (batch, seq, embed_dim)

        outputs = self.output(x)
        outputs_r, outputs_i = outputs.chunk(2, dim=-1)
        precoding_matrix = torch.complex(outputs_r, outputs_i)
        precoding_matrix = precoding_matrix.permute(0, 2, 1)

        return precoding_matrix


# Configs prédéfinies
CONFIGS = {
    "micro": dict(embedding_dim=32, num_heads=2, hidden_dim=128, num_layers=2),
    "tiny":  dict(embedding_dim=64, num_heads=4, hidden_dim=512, num_layers=4),
}