"""Prediction residual and the identity-predictor cosine control."""
import torch
from torch import nn
from torch.nn import functional as F


class Predictor(nn.Module):
    def __init__(self, dim: int, hidden: int = 1536):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, compressed):
        return compressed + self.net(compressed)


def scores(original, compressed, predictor):
    original = F.normalize(original, dim=-1)
    compressed = F.normalize(compressed, dim=-1)
    cosine = 1 - (original * compressed).sum(dim=-1)
    predicted = F.normalize(predictor(compressed), dim=-1)
    residual = ((original - predicted) ** 2).sum(dim=-1)
    return cosine, residual
