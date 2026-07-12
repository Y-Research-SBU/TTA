import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """Simple fusion head with selectable type.

    fusion_type: 'concat' (default), 'sum', 'mlp'
    in_dim: per-branch feature dim (D)
    out_dim: classifier output dim
    """

    def __init__(self, fusion_type: str = 'concat', in_dim: int = 256, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.head = nn.Linear(in_dim * 2, out_dim, bias=False)
        elif fusion_type == 'sum':
            self.head = nn.Linear(in_dim, out_dim, bias=False)
        elif fusion_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(in_dim, out_dim, bias=False)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, h_img: torch.Tensor, h_omic: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == 'concat':
            x = torch.cat([h_img, h_omic], dim=1)
            return self.head(x)
        elif self.fusion_type == 'sum':
            x = h_img + h_omic
            return self.head(x)
        elif self.fusion_type == 'mlp':
            x = torch.cat([h_img, h_omic], dim=1)
            x = self.mlp(x)
            return self.head(x)


