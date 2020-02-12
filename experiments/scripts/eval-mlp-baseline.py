"""Script for evaluation of MLP baseline models."""
import argparse
import importlib
import multiprocessing as mp
import os
import pickle
import yaml

import ge
from gem.embedding import sdne
import networkx as nx
import node2vec as n2v
import numpy as np
from sklearn import preprocessing as sk_prep
import torch
import torch.nn as nn
from torch.utils import data as tdata
from tqdm import tqdm

from attre2vec import utils as ae_utils


def get_args() -> argparse.Namespace:
    """Gets arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-cc',
        '--common-config',
        required=True,
        help='Path to common config file',
    )
    parser.add_argument(
        '-m',
        '--method',
        required=True,
        help='The used node embedding method',
    )
    parser.add_argument(
        '-d',
        '--dim',
        required=True,
        help='The node embedding method dimensionality',
    )
    return parser.parse_args()


def make_clf(path, args):
    """Creates a classifier given config."""
    module, model = path.rsplit('.', maxsplit=1)
    clf_class = getattr(importlib.import_module(module), model)

    clf = clf_class(**args)
    return clf


def train_model(model_name, model_dim, graph, is_pubmed):
    """Trains given model and returns node embeddings."""
    model_fns = {
        'dw': lambda g, d: train_node2vec(graph=g, dim=d, p=1, q=1),
        'n2v': lambda g, d: train_node2vec(graph=g, dim=d, p=4, q=1),
        'sdne': lambda g, d: train_sdne(graph=g, dim=d, is_pubmed=is_pubmed),
        'struc2vec': lambda g, d: train_struc2vec(graph=g, dim=d),
    }

    if model_name not in model_fns.keys():
        raise KeyError(f'No such model: \"{model_name}\"')

    return model_fns[model_name](graph, model_dim)


def train_node2vec(graph, dim, p, q):
    """Obtains node embeddings using Node2vec."""
    emb = n2v.Node2Vec(
        graph=graph,
        dimensions=dim,
        workers=mp.cpu_count(),
        p=p,
        q=q,
        quiet=True,
    ).fit()

    emb = {
        node_id: emb.wv[str(node_id)]
        for node_id in sorted(graph.nodes())
    }
    return emb


def train_sdne(graph, dim, is_pubmed):
    """Obtains node embeddings using SDNE."""
    mod = sdne.SDNE(
        d=dim,
        beta=5, alpha=1e-5,
        nu1=1e-6, nu2=1e-6,
        K=3, n_units=[50, 15],
        n_iter=200 if not is_pubmed else 50, xeta=0.01,
        n_batch=500,
    )
    mod.learn_embedding(graph=graph, is_weighted=False)
    np_emb = mod.get_embedding()

    emb = {
        node_id: np_emb[node_id]
        for node_id in sorted(graph.nodes())
    }
    return emb


def train_struc2vec(graph, dim):
    """Obtains node embedding using Struc2Vec."""
    g = nx.relabel_nodes(G=graph, mapping={n: str(n) for n in graph.nodes()})
    model = ge.Struc2Vec(g, workers=1, verbose=0)
    model.train(embed_size=dim, window_size=5, iter=5)
    e = model.get_embeddings()

    emb = {
        node_id: e[str(node_id)]
        for node_id in sorted(graph.nodes())
    }
    return emb


class SimpleMLPDataset(tdata.Dataset):
    """Implements a dataset used to train the MLP."""

    def __init__(self, ds, idx, tt, node_embs, use_edge_fts):
        """Inits SimpleMLPDataset."""
        if use_edge_fts:
            H = sk_prep.StandardScaler().fit_transform(ds['H'].numpy())
            self.X = torch.stack([
                torch.cat([
                    torch.tensor(node_embs[u]),
                    torch.tensor(node_embs[v]),
                    torch.tensor(H[ds['edge2idx'][(u, v)]])
                ])
                for u, v in ds['Xy'][idx][tt]['X']
            ])
        else:
            self.X = torch.stack([
                torch.cat([
                    torch.tensor(node_embs[u]),
                    torch.tensor(node_embs[v])
                ])
                for u, v in ds['Xy'][idx][tt]['X']
            ])
        self.y = torch.tensor(ds['Xy'][idx][tt]['y'])

    def __len__(self):
        """Returns the number of instances in the dataset."""
        return self.X.size(0)

    def __getitem__(self, idx):
        """Returns a single instance (x, y)."""
        return self.X[idx], self.y[idx]


class NodeMLP(nn.Module):
    """Implements a two-input MLP (node embeddings of two nodes)."""

    def __init__(self, dim):
        """Inits NodeMLP."""
        super().__init__()

        self._enc = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
        )
        self._dec = nn.Sequential(
            nn.Linear(dim, 2 * dim)
        )

    def forward(self, x):
        """Performs forward pass."""
        hid = self._enc(x)
        x_rec = self._dec(hid)
        return hid, x_rec


class NodeEdgeMLP(nn.Module):
    """Implements a three-input MLP (two node embeddings + edge features)."""

    def __init__(self, dim):
        """Inits NodeEdgeMLP."""
        super().__init__()

        self._enc = nn.Sequential(
            nn.Linear(2 * dim + 260, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
        )
        self._dec = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, 2 * dim + 260),
        )

    def forward(self, x):
        """Performs the forward pass."""
        hid = self._enc(x)
        x_rec = self._dec(hid)
        return hid, x_rec


def train_mlp_model(ds, idx, node_embs, num_epochs, use_edge_fts, dim):
    """Trains a given MLP model."""
    train_loader = tdata.DataLoader(
        dataset=SimpleMLPDataset(
            ds=ds,
            idx=idx,
            tt='train',
            node_embs=node_embs,
            use_edge_fts=use_edge_fts,
        ),
        batch_size=32,
        shuffle=True,
    )

    model = NodeEdgeMLP(dim=dim) if use_edge_fts else NodeMLP(dim=dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    mean_train_losses = []

    for i in range(num_epochs):
        train_losses = []
        for x_true, y_true in train_loader:
            optimizer.zero_grad()

            _, x_pred = model(x_true)
            loss = loss_fn(x_pred, x_true)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_losses.append(np.mean(train_losses))

    return model


def main():
    """Runs the script."""
    args = get_args()

    with open(args.common_config, 'r') as fin:
        common_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    dataset_path = common_cfg['paths']['input']['dataset']
    vectors_path = common_cfg['paths']['output']['vectors']
    metrics_path = common_cfg['paths']['output']['metrics']

    clf_cfg = common_cfg['base_clf']

    SCENARIOS = [
        {'name': 'MLP2', 'use_edge_fts': False},
        {'name': 'MLP3', 'use_edge_fts': True},
    ]

    with open(dataset_path, 'rb') as fin:
        ds = pickle.load(fin)

    G = ds['original_graph']

    if args.method == 'graphsage':
        node_embs = None
    else:
        node_embs = train_model(
            model_name=args.method,
            model_dim=int(args.dim),
            graph=G,
            is_pubmed='pubmed' in dataset_path
        )

    for scenario in tqdm(SCENARIOS, desc='Scenarios'):
        metrics = []

        for ds_idx in tqdm(range(ds['num_datasets']),
                           desc='Datasets',
                           leave=False):
            if args.method == 'graphsage':
                ds_name = (
                    dataset_path
                    .replace('data/datasets/', '')
                    .replace('.pkl', '')
                )
                node_embs = np.load(f'data/graphsage/{ds_name}-{ds_idx}.npy')

            mlp_model = train_mlp_model(
                ds=ds,
                idx=ds_idx,
                node_embs=node_embs,
                num_epochs=100,
                use_edge_fts=scenario['use_edge_fts'],
                dim=int(args.dim),
            )

            with torch.no_grad():
                X = {
                    tt: mlp_model(
                        SimpleMLPDataset(
                            ds=ds, idx=ds_idx,
                            tt=tt, node_embs=node_embs,
                            use_edge_fts=scenario['use_edge_fts'],
                        ).X
                    )[0].numpy()
                    for tt in ('train', 'val', 'test')
                }
                y = {tt: ds['Xy'][ds_idx][tt]['y'] for tt in
                     ('train', 'val', 'test')}

            # Train
            model = make_clf(path=clf_cfg['module'], args=clf_cfg['args'])
            model.fit(X['train'], y['train'])

            # Eval
            ds_metrics = {}
            for tt in ('train', 'val', 'test'):
                ds_metrics[tt] = ae_utils.calc_metrics(
                    y_score=model.predict_proba(X[tt]),
                    y_pred=model.predict(X[tt]),
                    y_true=y[tt],
                    max_cls=ds['num_cls'],
                )

            metrics.append(ds_metrics)

            # Save vectors
            vecs = [
                *X['train'].tolist(),
                *X['val'].tolist(),
                *X['test'].tolist(),
            ]
            vp = (
                vectors_path
                .replace(
                    '${NAME}',
                    'MLP_' + args.method + '/' + scenario['name'],
                )
                .replace('${IDX}', str(ds_idx))
            )
            os.makedirs(os.path.dirname(vp), exist_ok=True)

            with open(vp, 'wb') as fout:
                pickle.dump(vecs, fout)

        # Save
        mtrp = metrics_path.replace(
            '${NAME}',
            args.method + '/' + scenario['name'],
        )
        os.makedirs(os.path.dirname(mtrp), exist_ok=True)
        with open(mtrp, 'wb') as fout:
            pickle.dump(metrics, fout)


if __name__ == '__main__':
    main()
