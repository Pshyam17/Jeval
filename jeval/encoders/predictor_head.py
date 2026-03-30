from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch is not None and nn is not None:
    class PreLNTransformerPredictor(nn.Module):
        """Predictor head that maps encoder embeddings to original encoder space via Pre-LN transformer."""

        def __init__(self, dim: int, hidden_dim: int = 512, n_heads: int = 8, num_layers: int = 3):
            super().__init__()
            if dim <= 0:
                raise ValueError("dim must be positive")
            self.input_proj = nn.Linear(dim, hidden_dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,  # Pre-LN
                )
                for _ in range(num_layers)
            ])
            self.output_proj = nn.Linear(hidden_dim, dim)

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            if x.ndim == 2:
                x = x.unsqueeze(1)

            if x.ndim != 3:
                raise ValueError("Predictor input must be 2D or 3D tensor")

            x = self.input_proj(x)
            for layer in self.layers:
                x = layer(x)

            x = self.output_proj(x)
            if x.shape[1] == 1:
                x = x.squeeze(1)
            return F.normalize(x, dim=-1)
else:
    class PreLNTransformerPredictor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Torch is required for PreLNTransformerPredictor")
    """Predictor head that maps encoder embeddings to original encoder space via Pre-LN transformer."""

    def __init__(self, dim: int, hidden_dim: int = 512, n_heads: int = 8, num_layers: int = 3):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, dim)

        # Xavier initialization supports stable transformer convergence.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [batch, seq, dim] or [batch, dim] if single token embedding."""
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise ValueError("Predictor input must be 2D or 3D tensor")

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_proj(x)
        # For EPE we compare with single embedding; reduce seq dim if needed.
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x
