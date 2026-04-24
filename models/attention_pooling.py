import torch
import torch.nn as nn


class AttentionPooling(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled
