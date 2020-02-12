"""Utility functions for aggregators."""
from argparse import Namespace

from attre2vec.aggregators import avg
from attre2vec.aggregators import base
from attre2vec.aggregators import rnn


def make_aggregator(name: str, hparams: Namespace) -> base.BaseAggregator:
    """Creates aggregator object."""
    aggregators = {
        'SimpleAverageAggregator': (
            lambda hp: avg.SimpleAverageAggregator()
        ),
        'ExponentialAverageAggregator': (
            lambda hp: avg.ExponentialAverageAggregator()
        ),

        'ConcatGRUAggregator': (
            lambda hp: rnn.ConcatGRUAggregator(
                edge_dim=hp.dims_edge,
                node_dim=hp.dims_node,
            )
        ),
        'GRUAggregator': (
            lambda hp: rnn.GRUAggregator(edge_dim=hp.dims_edge)
        ),
    }

    if name not in aggregators.keys():
        raise RuntimeError(f'No such aggregator: \"{name}\"')

    return aggregators[name](hparams)
