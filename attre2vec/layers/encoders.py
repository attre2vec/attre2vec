"""Implementations of encoder models."""
from argparse import Namespace

import torch
from torch import nn


def make_encoder(name: str, hparams: Namespace):
    """Creates encoder object."""
    encoders = {
        'Encoder': (
            lambda hp: Encoder(
                edge_dim=hp.dims_edge,
                output_dim=hp.dims_emb,
            )
        ),
    }

    if name not in encoders.keys():
        raise RuntimeError(f'No such encoder: \"{name}\"')

    return encoders[name](hparams)


class Attention(nn.Module):
    """Implements attention mechanism for multi input MLP."""

    def __init__(self, num_inputs, input_dim):
        """Inits Attention."""
        super().__init__()
        self._ahs = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 1), nn.Tanh())
            for _ in range(num_inputs)
        ])
        self._sm = nn.Softmax(dim=-1)

    def forward(self, *inputs):
        """Calculates score for each input and feeds them into softmax."""
        scores = [ah(inp) for ah, inp in zip(self._ahs, inputs)]
        alpha = self._sm(torch.cat(scores, dim=-1)).unsqueeze(-1)
        return alpha


class Encoder(nn.Module):
    """Encodes edge emb and flows from nodes into common representation."""

    def __init__(self, edge_dim, output_dim):
        """Inits Encoder."""
        super().__init__()
        num_inputs = 3

        self._layers = nn.Sequential(
            nn.Linear(num_inputs * edge_dim, 2 * edge_dim),
            nn.ReLU(),
            nn.Linear(2 * edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, output_dim),
            nn.Tanh(),
        )
        self._att = Attention(num_inputs=num_inputs, input_dim=edge_dim)

    def forward(self, h_t_1, f_u, f_v):
        """Implements forward pass of model."""
        alpha = self._att(h_t_1, f_u, f_v)

        x = alpha * torch.stack([h_t_1, f_u, f_v], dim=1)

        h_t = self._layers(torch.cat(
            [x[:, 0], x[:, 1], x[:, 2]],
            dim=-1
        ))

        return h_t, alpha
