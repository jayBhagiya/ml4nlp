import math

import torch
from torch import nn
from torch_geometric.utils import scatter, softmax


def complete_edge_index(batch: torch.Tensor) -> torch.Tensor:
    """Return directed, loop-free complete edges within each batched graph."""
    if batch.ndim != 1 or batch.numel() == 0:
        raise ValueError("batch must be a non-empty one-dimensional tensor")

    graphs = []
    for graph_id in range(int(batch.max()) + 1):
        nodes = torch.where(batch == graph_id)[0]
        if nodes.numel() < 2:
            continue
        pairs = torch.combinations(nodes, r=2)
        graphs.append(torch.cat((pairs, pairs.flip(1))).t())

    if not graphs:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(graphs, dim=1)


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        shape = (-1, self.heads, self.head_dim)
        query = self.query(x).view(shape)
        key = self.key(x).view(shape)
        value = self.value(x).view(shape)

        scores = (query[target] * key[source]).sum(-1) / math.sqrt(self.head_dim)
        attention = softmax(scores.clamp(-5, 5), target, num_nodes=x.size(0))
        messages = value[source] * attention.unsqueeze(-1)
        output = scatter(
            messages,
            target,
            dim=0,
            dim_size=x.size(0),
            reduce="sum",
        )
        return output.flatten(1), attention


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        normalization: str,
        dropout: float,
    ):
        super().__init__()
        if normalization not in {"batch", "layer"}:
            raise ValueError("normalization must be 'batch' or 'layer'")

        normalization_layer = nn.BatchNorm1d if normalization == "batch" else nn.LayerNorm
        self.attention = MultiHeadGraphAttention(hidden_dim, heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = normalization_layer(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = normalization_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(x, edge_index)
        x = self.norm1(x + self.output(self.dropout(attended)))
        x = self.norm2(x + self.feed_forward(x))
        return x, weights


class GraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        classes: int = 2,
        hidden_dim: int = 80,
        heads: int = 8,
        layers: int = 10,
        positional_dim: int = 2,
        use_laplacian_pe: bool = True,
        full_attention: bool = False,
        normalization: str = "batch",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.positional_dim = positional_dim
        self.use_laplacian_pe = use_laplacian_pe
        self.full_attention = full_attention
        self.node_embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        if use_laplacian_pe:
            self.position_embedding = nn.Linear(positional_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GraphTransformerLayer(hidden_dim, heads, normalization, dropout)
            for _ in range(layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        laplacian_pe: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ):
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"x must have shape [nodes, {self.input_dim}]")
        if batch.ndim != 1 or batch.size(0) != x.size(0):
            raise ValueError("batch must contain one graph index per node")

        x = self.node_embedding(x.float())
        if self.use_laplacian_pe:
            if laplacian_pe is None:
                raise ValueError("laplacian_pe is required for this model")
            if laplacian_pe.shape != (x.size(0), self.positional_dim):
                raise ValueError(
                    f"laplacian_pe must have shape [nodes, {self.positional_dim}]"
                )
            if self.training:
                graph_count = int(batch.max()) + 1
                signs = torch.randint(
                    0,
                    2,
                    (graph_count, self.positional_dim),
                    device=x.device,
                    dtype=torch.long,
                ).mul_(2).sub_(1)
                laplacian_pe = laplacian_pe * signs[batch]
            x = x + self.position_embedding(laplacian_pe.float())

        if self.full_attention:
            edge_index = complete_edge_index(batch)
        else:
            source, target = edge_index
            edge_index = edge_index[:, source != target]

        last_attention = None
        for layer in self.layers:
            x, last_attention = layer(x, edge_index)
        logits = self.readout(x)

        if return_attention:
            return logits, edge_index, last_attention
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

