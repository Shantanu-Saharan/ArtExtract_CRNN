import torch
import torch.nn as nn
from torchvision import models


class ResNetCRNN(nn.Module):

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 512,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        self.cnn = backbone.features

        self.feature_dim = 1536
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=8, dropout=0.1, batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor):
        features = self.cnn(x)
        B, C, H, W = features.shape
        sequence = features.view(B, C, H * W).permute(0, 2, 1)

        lstm_out, _ = self.lstm(sequence)

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)

        logits = self.head(pooled)

        return logits, pooled
