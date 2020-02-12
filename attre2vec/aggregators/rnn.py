"""Implementations of edge walk aggregators."""
import torch

from attre2vec.aggregators import base
from attre2vec import rnn


class GRUAggregator(base.BaseAggregator):
    """Aggregation using simple GRU cells."""

    def __init__(self, edge_dim):
        """Inits GRUAggregator."""
        super().__init__()
        self._gru = rnn.GRUCell(num_inputs=edge_dim, num_hidden=edge_dim)

    def aggregate(
        self,
        edge_features: torch.Tensor,
        nodes_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates single edge walk into feature vector."""
        batch_size, num_walks, walk_length, _ = edge_features.size()
        bs_nw = batch_size * num_walks

        h = None
        # Reversed because we want to aggregate features starting from the
        # furthest to the nearest edge.
        for idx in reversed(range(walk_length)):
            _, h = self._gru(
                x=edge_features[:, :, idx, :].view(bs_nw, -1),
                h_t_1=h,
            )

        h = h.view(batch_size, num_walks, -1)
        return h


class ConcatGRUAggregator(base.BaseAggregator):
    """Aggregation using GRU cells with input as edge and node concat."""

    def __init__(self, edge_dim, node_dim):
        """Inits ConcatGRUAggregator."""
        super().__init__()
        self._gru = rnn.GRUCell(
            num_inputs=edge_dim + node_dim,
            num_hidden=edge_dim,
        )

    def aggregate(
        self,
        edge_features: torch.Tensor,
        nodes_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates single edge walk into feature vector."""
        batch_size, num_walks, walk_length, _ = edge_features.size()
        bs_nw = batch_size * num_walks

        h = None
        # Reversed because we want to aggregate features starting from the
        # furthest to the nearest edge.
        for idx in reversed(range(walk_length)):
            _, h = self._gru(
                x=torch.cat([
                    edge_features[:, :, idx, :].view(bs_nw, -1),
                    nodes_features[:, :, idx, :].view(bs_nw, -1)
                ], dim=1),
                h_t_1=h,
            )

        h = h.view(batch_size, num_walks, -1)
        return h
