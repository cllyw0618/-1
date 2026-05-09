import torch
import torch.nn as nn
from torch import Tensor

from model.EnhancedHGNNConv import EnhancedHGNNConv


class EnhancedHGNN(nn.Module):
    """Enhanced HGNN encoder used by the final method."""

    def __init__(
        self,
        node_dim: int,
        emb_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, emb_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                EnhancedHGNNConv(
                    node_in_dim=emb_dim,
                    node_out_dim=emb_dim,
                    hyper_in_dim=emb_dim,
                    hyper_out_dim=emb_dim,
                    drop_rate=0.1,
                    is_last=(i == num_layers - 1),
                    num_heads=num_heads,
                    attn_dim=attn_dim,
                )
            )

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> tuple[Tensor, Tensor]:
        x = self.node_encoder(x)
        y = self._init_hyper_emb(x, hyperedge_index)
        for layer in self.layers:
            x, y = layer(x, y, hyperedge_index)
        return x, y

    @staticmethod
    def _init_hyper_emb(x: Tensor, hyperedge_index: Tensor) -> Tensor:
        node_idx, edge_idx = hyperedge_index
        num_hyperedges = int(edge_idx.max().item()) + 1

        counts = torch.zeros(num_hyperedges, device=x.device)
        counts.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.float))
        counts = torch.clamp(counts, min=1)

        y = torch.zeros(num_hyperedges, x.size(1), device=x.device)
        y.scatter_add_(0, edge_idx.unsqueeze(1).expand(-1, x.size(1)), x[node_idx])
        y = y / counts.unsqueeze(1)
        return y
