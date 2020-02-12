"""Code for edge sample&aggregate layers."""
import torch

from attre2vec import aggregators as ng_agg
from attre2vec.layers import encoders as ng_enc


class EdgeSampleAggregate(torch.nn.Module):
    """Implements edge-based sample and aggregate model."""

    @classmethod
    def from_cfg(cls, hparams):
        """Creates AttrE2vec object from configuration."""
        # Aggregator
        aggregator = ng_agg.make_aggregator(
            name=hparams.aggregator,
            hparams=hparams,
        )

        # Encoder
        encoder = ng_enc.make_encoder(
            name=hparams.encoder,
            hparams=hparams,
        )

        return cls(
            edge_dim=hparams.dims_edge,
            node_dim=hparams.dims_node,
            aggregator=aggregator,
            encoder=encoder,
        )

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        aggregator: ng_agg.BaseAggregator,
        encoder: torch.nn.Module,
    ):
        """Inits EdgeSampleAggregate."""
        super().__init__()

        self._edge_dim = edge_dim
        self._node_dim = node_dim

        self._agg = aggregator
        self._enc = encoder

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._agg.to(self._device)
        self._enc.to(self._device)

    def forward(self, xe, h_xe, h, m, rws_idx):
        """Performs forward pass of model."""
        w = {'u': {'edge': [], 'node': []}, 'v': {'edge': [], 'node': []}}
        for edge in xe.tolist():
            for key, node in (('u', edge[0]), ('v', edge[1])):
                idxs = rws_idx[tuple(edge)][node]
                k, L = idxs['edge'].shape
                w[key]['edge'].append(h[idxs['edge'].view(-1)].view(k, L, -1))
                w[key]['node'].append(m[idxs['node'].view(-1)].view(k, L, -1))

        w['u'] = {k: torch.stack(v).to(self._device) for k, v in w['u'].items()}
        w['v'] = {k: torch.stack(v).to(self._device) for k, v in w['v'].items()}

        # Aggregate
        fs = {
            key: self._agg.aggregate(
                edge_features=w[key]['edge'],
                nodes_features=w[key]['node'],
            )
            for key in ('u', 'v')
        }

        # Aggregate across multiple random walks
        f = {key: fs['u'].mean(dim=1) for key in ('u', 'v')}

        # Encode
        h_xe_new, alpha = self._enc(h_t_1=h_xe, f_u=f['u'], f_v=f['v'])

        # Training related metadata
        metadata = {
            'alpha': alpha,
        }

        return h_xe_new, metadata
