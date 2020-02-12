"""Implementation of decoders for unsupervised model."""
from argparse import Namespace

from torch import nn


def make_decoder(name: str, hparams: Namespace):
    """Creates encoder object."""
    decoders = {
        'MLPDecoder': (
            lambda hp: MLPDecoder(
                emb_dim=hp.dims_emb,
                edge_dim=hp.dims_edge,
            )
        ),
    }

    if name not in decoders.keys():
        raise RuntimeError(f'No such decoder: \"{name}\"')

    return decoders[name](hparams)


class MLPDecoder(nn.Module):
    """Takes edge embedding and tries to reconstruct original features."""

    def __init__(self, emb_dim, edge_dim):
        """Inits EdgeClassifier."""
        super().__init__()
        hidden_dim = (emb_dim + edge_dim) // 2

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, h_t):
        """Implements forward pass of model."""
        return self.layers(h_t)
