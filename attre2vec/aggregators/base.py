"""Implementations of edge walk aggregators."""
import abc

import torch
from torch import nn


class BaseAggregator(abc.ABC, nn.Module):
    """Base class for edge walk aggregators."""

    def __init__(self):
        """Inits BaseAggregator."""
        super().__init__()
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    @abc.abstractmethod
    def aggregate(
        self,
        edge_features: torch.Tensor,
        nodes_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates single edge walk into feature vector."""
        pass
