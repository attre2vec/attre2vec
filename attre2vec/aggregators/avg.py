"""Implementations of edge walk aggregators."""
import abc

import numpy as np
import torch

from attre2vec.aggregators import base


class ConstantWeightsAggregator(base.BaseAggregator):
    """Uses provided weights to calculate weighted average of features."""

    def aggregate(
        self,
        edge_features: torch.Tensor,
        nodes_features: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregates single edge walk into feature vector."""
        batch_size, num_walks, walk_length, edge_dim = edge_features.size()

        weights = (
            self
            .get_weights(walk_length)
            .repeat(num_walks, 1)
            .reshape(1, num_walks, walk_length, 1)
        )

        weighted = torch.mul(weights, edge_features)
        aggregated = torch.sum(weighted, dim=2) / walk_length

        return aggregated

    @abc.abstractmethod
    def get_weights(self, walk_length: int) -> torch.Tensor:
        """Calculates weights for feature vectors."""
        pass


class SimpleAverageAggregator(ConstantWeightsAggregator):
    """Implementation of aggregation using simple average."""

    def get_weights(self, walk_length: int) -> torch.Tensor:
        """Calculates weights for feature vectors."""
        return torch.ones(
            size=(walk_length,),
            dtype=torch.float,
            device=self._device,
        )


class ExponentialAverageAggregator(ConstantWeightsAggregator):
    """Naive implementation of aggregation using exponential decaying."""

    def get_weights(self, walk_length: int) -> torch.Tensor:
        """Calculates weights for feature vectors."""
        return torch.tensor(
            [np.exp(-i) for i in range(walk_length)],
            dtype=torch.float,
            device=self._device,
        )
